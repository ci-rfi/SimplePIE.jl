module SimplePIE

using Configurations

using Unitful
using Unitful: Å, nm, μm, °, kV, mrad
using MAT
using Statistics: mean
using FFTW
using Images
using Plots
using DSP: unwrap
using IterTools
using Random
using AxisArrays
using ThreadsX
using CUDA
using BenchmarkTools
using HDF5

using Dates

export wavelength
export circular_aperture
export define_probe_positions
export make_object
export make_probe
export load_dps
# export update!
export ptycho_iteration!
export gpu_ptycho_iteration!
export plot_wave
export ptycho_reconstruction!
export plot_amplitude
export plot_phase

export load_mib

abstract type AbstractMIBHeader end

struct MIBHeader <: AbstractMIBHeader
    id::Int
    offset::Int
    nchip::Int
    dims::Vector{Int}
    data_type::DataType
    chip_dims::Vector{Int}
    time::DateTime
    exposure_s::Float64
    image_bit_depth::Int
    raw::Bool
end

function load_mib(filepath::AbstractString; kwargs...)
    first_header = firstheader(filepath)
    images, headers = read_mib(filepath, first_header; kwargs...)
    return images, headers
end

function read_mib(filepath::AbstractString, first_header::AbstractMIBHeader; range=[1,typemax(Int)])
    offset = first_header.offset
    type = first_header.data_type
    dims = first_header.dims
    raw = first_header.raw
    image_bit_depth = first_header.image_bit_depth

    fid = open(filepath, "r")
    headers = Vector{MIBHeader}()
    if raw
        depth_dict = Dict(1 => UInt8, 6 => UInt8, 12 => UInt16,
                          24 => UInt32, 48 => UInt64)
        type = depth_dict[image_bit_depth]
    end
        buffer = Array{type}(undef, dims[1], dims[2])
        images = Vector{Array{type, 2}}()

    n = 0
    while eof(fid) == false && n < range[2]
            header_string = read(fid, offset)
            read!(fid, buffer)
            n += 1
            if n >= range[1]
                push!(headers, make_mibheader(String(header_string); id=n))
                push!(images, hton.(buffer))
            end
    end
    close(fid)
    return images, headers
end

function firstheader(filepath)
    fid = open(filepath)
    trial = split(String(read(fid, 768)), ",")
    offset = parse(Int, trial[3])
    seekstart(fid)
    header_string = String(read(fid, offset))
    close(fid)
    first_header = make_mibheader(header_string; id=1)
    return first_header
end

function make_mibheader(header_string::AbstractString; id=0)
    header = split(header_string, ",")
    offset = parse(Int, header[3])
    nchip = parse(Int, header[4])
    dims = parse.(Int, header[5:6])
    type_dict = Dict("U1" => UInt8, "U8" => UInt8, "U08" => UInt8, "U16" => UInt16,
                     "U32" => UInt32, "U64" => UInt64, "R64" => UInt64)
    data_type = type_dict[header[7]]
    chip_dims = parse.(Int, split(lstrip(header[8]), "x"))
    time = DateTime(header[10][1:end-3], "y-m-d H:M:S.s")
    exposure_s = parse(Float64, header[11])
    image_bit_depth = parse(Int, header[end-1])
    raw = header[7] == "R64"
    return MIBHeader(id, offset, nchip, dims, data_type, chip_dims, time, exposure_s, image_bit_depth, raw)
end

function wavelength(V)::typeof(1.0u"nm")
    e  = 1.60217663e-19u"C" 
    m₀ = 9.10938370e-31u"kg" 
    c  = 2.99792458e8u"m/s" 
    h  = 6.62607015e-34u"N*m*s" 
    λ  = h / sqrt(2m₀ * e * V * (1 + e * V / (2m₀ * c^2)))
    return λ
end

function circular_aperture(n::Integer, r; shift=CartesianIndex(0, 0), σ=0)
    data = Matrix{Bool}(undef, n, n)
    if n <= 2r 
        @warn("Aperature area exceeds the field of view even if centered.") 
    end
    origin =  CartesianIndex(ceil.(Int, size(data) ./ 2)...) + shift
    for ind in CartesianIndices(data)
        data[ind] = hypot(Tuple(ind - origin)...) <= r ? true : false
    end
    aperture = imfilter(data, Kernel.gaussian(σ))
    return aperture
end

function define_probe_positions(dₛ, θᵣ, n₁, n₂; offset=[zero(dₛ), zero(dₛ)])
    init_positions = [[(cos(θᵣ)j - sin(θᵣ)i)dₛ, (cos(θᵣ)i + sin(θᵣ)j)dₛ] for (i,j) in product(1:n₁, 1:n₂)]
    min_x = minimum(first, init_positions)
    min_y = minimum(last, init_positions)
    positions = map(init_positions) do p
        p .- [min_x, min_y] .+ offset
    end
    return positions 
end
define_probe_positions(dₛ, θᵣ, n; kwargs...) = define_probe_positions(dₛ, θᵣ, n, n; kwargs...)

function make_object(positions, N, Δx, Δy; data_type=ComplexF32)
    min_x = minimum(first, positions)
    min_y = minimum(last, positions)
    max_x = maximum(first, positions)
    max_y = maximum(last, positions)

    padding_x = 0.5(N+1) * Δx
    padding_y = 0.5(N+1) * Δy

    𝒪_min_x = min_x - padding_x
    𝒪_min_y = min_y - padding_y 
    𝒪_max_x = max_x + padding_x
    𝒪_max_y = max_y + padding_y 

    nx = length(𝒪_min_x:Δx:𝒪_max_x)
    ny = length(𝒪_min_y:Δy:𝒪_max_y)

    𝒪 = AxisArray(ones(data_type, nx,ny); x = (𝒪_min_x:Δx:𝒪_max_x), y = (𝒪_min_y:Δy:𝒪_max_y))
    ℴ = map(positions) do p
        x1 = p[1] - Δx*N/2
        x2 = p[1] + Δx*N/2
        y1 = p[2] - Δy*N/2
        y2 = p[2] + Δy*N/2
        view(𝒪, x1 .. x2, y1 .. y2)
    end
    return 𝒪, ℴ
end
make_object(positions, N, Δx; kwargs...) = make_object(positions, N, Δx, Δx; kwargs...)

function make_probe(α, N, Δf, Δk, Δx; data_type=ComplexF32, mean_amplitude_sum=1)
    K = [Δk * [i,j] for (i,j) in product(-N/2:N/2-1, -N/2:N/2-1)]
    ω = map(x -> x[1] + x[2]im, K)
    ωᵢ = map(x -> x[1] - x[2]im, K)
    # ϕ = map(x -> atan(x...), K)
    χ = (ω .* ωᵢ) * Δf / 2 |> real |> x -> uconvert.(nm, x)
    aberration = -2π/λ * χ 
    aperture = circular_aperture(N, Int(round(α/Δk)); σ=1) 
    𝒟 = cis.(aberration) .* aperture 
    𝒟 = 𝒟 / sum(abs.(𝒟)) * mean_amplitude_sum

    𝒫_array = fftshift(ifft(ifftshift(𝒟))) |> Matrix{data_type}
    𝒫_min_x = -0.5(N+1) * Δx
    𝒫_max_x = 0.5(N-2) * Δx
    𝒫 = AxisArray(𝒫_array; x = (𝒫_min_x:Δx:𝒫_max_x), y = (𝒫_min_x:Δx:𝒫_max_x))
    return 𝒫
end

function load_dps(filename, n₁, n₂)
    dps_mat = matread(filename)["dps"];
    dps = [dps_mat[:,:, i + (j-1)*n₁] for (i,j) in product(1:n₁, 1:n₂)]
    return dps
end
load_dps(filename, n) = load_dps(filename, n, n)

function update!(q, a, Δψ; method="ePIE", α=0.2) 
    a̅ = conj(a)
    aₘ = maximum(abs, a)
    aₘ² = aₘ^2

    if method == "ePIE"
        # w = a̅ / aₘ²
        w = α * a̅ / aₘ²
    #TODO: Test rPIE (not working at the moment)
    elseif method == "rPIE"
        a² = a .^ 2
        w = a̅ ./ ((1 - α) * a² .+ α * aₘ²)
    #TODO: Test PIE (not working at the moment)
    elseif method == "PIE"
        a² = a .^ 2
        w = (abs.(a) / aₘ) .* (a̅ ./ (a² .+ α * aₘ²))
    else
        error("$method is not a supported update method.")
    end

    q[:] = q + w .* Δψ
    return nothing
end

function ptycho_iteration!(𝒪, 𝒫, 𝒜 ; method="ePIE", α=0.2, β=0.2)
    ψ₁ = 𝒪 .* 𝒫
    𝒟 = 𝒜 .* sign.(fft(ψ₁))
    ψ₂ = ifft(𝒟)
    Δψ = ψ₂ - ψ₁
    update!(𝒪, 𝒫, Δψ; method=method, α=α)
    update!(𝒫, 𝒪, Δψ; method=method, α=β)
    return nothing
end

function gpu_ptycho_iteration!(𝒪_cpu, 𝒫_cpu, 𝒜_cpu; method="ePIE", α::Float32=Float32(0.2), β::Float32=Float32(0.2))
    𝒪 = CuArray(copy(𝒪_cpu.data))
    𝒫 = CuArray(copy(𝒫_cpu.data))
    𝒜 = CuArray(𝒜_cpu)
    ptycho_iteration!(𝒪, 𝒫, 𝒜; method=method, α=α, β=β)
    copyto!(𝒪_cpu, Array(𝒪))
    copyto!(𝒫_cpu, Array(𝒫))
    return nothing
end

function plot_amplitude(𝒲)
    amplitude = abs.(𝒲)
    return heatmap(amplitude, aspect_ratio=1)
end

function plot_phase(𝒲; unwrap_phase=false)
    phase = unwrap_phase ? unwrap(angle.(𝒲); dims=1:2) : angle.(𝒲)
    return heatmap(phase, aspect_ratio=1)
end

function plot_wave(𝒲; unwrap_phase=false)
    # amplitude = abs.(𝒲)
    # phase = unwrap_phase ? unwrap(angle.(𝒲); dims=1:2) : angle.(𝒲)
    # p1 = heatmap(amplitude, aspect_ratio=1)
    # p2 = heatmap(phase, aspect_ratio=1)
    p1 = plot_amplitude(𝒲)
    p2 = plot_phase(𝒲; unwrap_phase=false)
    return plot(p1, p2, layout=(1,2))
end

function ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, nᵢ; method="ePIE", α=Float32(0.01), β=Float32(0.01), ngpu::Integer=0, plotting=false)
    for _ in 1:nᵢ
        if ngpu == 0
            @time Threads.@threads for (i,j) in shuffle(collect(product(1:n, 1:n)))
                ptycho_iteration!(ℴ[i,j], 𝒫, 𝒜[i,j]; method=method, α=α, β=β)
            end
        else 
            ngpu = min(ngpu, CUDA.ndevices())
            @time Threads.@threads for (i,j) in shuffle(collect(product(1:n, 1:n)))
                CUDA.device!((i*n + j) % ngpu)
                gpu_ptycho_iteration!(ℴ[i,j], 𝒫, 𝒜[i,j]; method=method, α=α, β=β)
            end
        end

        if plotting
            display(plot_wave(𝒫))
            display(plot_wave(𝒪))
        end
    end
    return nothing
end

# output_file = "/home/chen/Data/ssd/2022-05-27/20220526_195851/rotation_search_1to360.h5"
function rotation_sweep(output_file, 𝒜, dₛ, n, N, Δx, α, Δf, Δk, mean_amplitude_sum; nᵢ=1, ngpu=4, sweep_range=1°:1°:360°, offset=[zero(dₛ), zero(dₛ)])
    for θᵣ = sweep_range
        positions = define_probe_positions(dₛ, θᵣ, n; offset=[offset, offset])
        𝒪, ℴ = make_object(positions, N, Δx) 
        𝒫 = make_probe(α, N, Δf, Δk, Δx; mean_amplitude_sum=mean_amplitude_sum)
        # ptycho_iteration_gpu(ℴ, 𝒫, 𝒜; nᵢ=nᵢ)
        ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, nᵢ; ngpu=ngpu, plotting=false)
        h5write(output_file, "/object" * string(lpad(ustrip(θᵣ),3,"0")), convert(Matrix{ComplexF32}, 𝒪))
        h5write(output_file, "/probe" * string(lpad(ustrip(θᵣ),3,"0")), convert(Matrix{ComplexF32}, 𝒫))
    end
    phase_max_min = map(1:360) do i 
        oo = h5read(output_file, "/object" * string(lpad(i,3,"0")))
        [maximum(angle.(oo)), minimum(angle.(oo))]
        end
    findmax(first.(phase_max_min) - last.(phase_max_min))
end
@option struct PtychoParams
    # N::Int
    # n::Int
    # λ::typeof(1.0nm) 
    # α::typeof(1.0mrad)
    # Δk::typeof(1.0mrad)
    # θ::typeof(1.0mrad)
    # θᵣ::typeof(1.0°)
    # dₛ::typeof(1.0Å)
    # Δx = uconvert(Å, λ/θ)
    # Δf::typeof(1.0μm)

    detector_array_size::Int
    scan_array_size::Int
    wave_legnth::typeof(1.0nm) 
    convergence_semi_angle::typeof(1.0mrad)
    fourier_space_sampling::typeof(1.0mrad)
    maximum_angle::typeof(1.0mrad)
    rotation_angle::typeof(1.0°)
    step_size::typeof(1.0Å)
    real_space_sampling::typeof(1.0Å)
    defocus::typeof(1.0μm)
end

# filename = "../AMaiden_ptycho_EM/dps_intensity_rotavirus_demo_data.mat"
# N = 256
# n = 127
# dps = load_dps(filename, n)
# λ = wavelength(300kV)
# α = 1.0340mrad
# D = 125
# Δk = 2α / D
# θ = N * Δk
# θᵣ = -126°
# dₛ = 31.25Å
# Δx = uconvert(Å, λ/θ)
# Δf = -13μm

offset = uconvert(Å, (0.5N+1) * Δx)
positions = define_probe_positions(dₛ, θᵣ, n; offset=[offset, offset])

𝒜 = ThreadsX.map(x -> fftshift(sqrt.(x)) |> Matrix{Float32}, dps)
𝒪, ℴ = make_object(positions, N, Δx) 
𝒫 = make_probe(α, N, Δf, Δk, Δx; mean_amplitude_sum=sum(sqrt.(mean(dps))))
plot_wave(𝒫; unwrap_phase=true)

# nᵢ = 3
# for _ in 1:nᵢ
#     @time Threads.@threads for (i,j) in shuffle(collect(product(1:n, 1:n)))
#         ptycho_iteration!(ℴ[i,j], 𝒫, 𝒜[i,j]; method="ePIE", α=0.01, β=0.01)
#     end
#     display(plot_wave(𝒫))
#     display(plot_wave(𝒪))
# end
ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, 3)

# nᵢ = 10
# for _ in 1:nᵢ
#     @time Threads.@threads for (i,j) in shuffle(collect(product(1:n, 1:n)))
#         CUDA.device!((i*n + j) % 4)
#         gpu_ptycho_iteration!(ℴ[i,j], 𝒫, 𝒜[i,j]; method="ePIE", α=Float32(0.01), β=Float32(0.01))
#     end
#     display(plot_wave(𝒫))
#     display(plot_wave(𝒪))
# end
ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, 10; ngpu=4, plotting=false)






# using HDF5
# output_file = "./results/rotavirus_ePIE_10iter.h5"
# h5write(output_file, "/object", convert(Matrix{ComplexF32}, 𝒪))
# h5write(output_file, "/probe", convert(Matrix{ComplexF32}, 𝒫))



end
