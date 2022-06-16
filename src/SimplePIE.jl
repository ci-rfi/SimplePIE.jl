module SimplePIE

using Configurations
using TOML
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
using Medipix

import Configurations.from_dict
import Configurations.to_dict

export PtychoParams
export ObjectParams
export ProbeParams
export IterParams
export SweepParams
export from_toml

export wavelength
export circular_aperture
export make_grid
export make_object
export sum_sqrt_mean
export make_probe
export load_dps
export load_mib
export make_amplitude
export ptycho_iteration!
export gpu_ptycho_iteration!
export plot_wave
export ptycho_reconstruction!
export plot_amplitude
export plot_phase
export save_object
export save_probe
export save_result
export crop_center
export cbed_center
export edge_distance
export shift_cbed
export align_cbeds
export rotation_angle_sweep
export stepsize_sweep
export defocus_sweep
export parameter_sweep

@option mutable struct PtychoParams
    detector_array_size::Int = 0
    scan_array_size::Vector{Int} = [0, 0]
    wavelength::typeof(1.0nm) = 0.0nm
    convergence_semi_angle::typeof(1.0mrad) = 0.0mrad
    fourier_space_sampling::typeof(1.0mrad) = 0.0mrad
    maximum_angle::typeof(1.0mrad) = 0.0mrad
    rotation_angle::typeof(1.0°) = 0.0°
    step_size::typeof(1.0Å) = 0.0Å
    real_space_sampling::typeof(1.0Å) = 0.0Å
    defocus::typeof(1.0μm) = 0.0μm
    amplitude_sum::Float64 = 0
end

@option mutable struct ObjectParams
    step_size::typeof(1.0Å) = 0.0Å
    rotation_angle::typeof(1.0°) = 0.0°
    scan_array_size::Vector{Int} = [0, 0]
    detector_array_size::Int = 0
    real_space_sampling::typeof(1.0Å) = 0.0Å
end
ObjectParams(p::PtychoParams) = ObjectParams(p.step_size, p.rotation_angle, p.scan_array_size, p.detector_array_size, p.real_space_sampling)

@option mutable struct ProbeParams
    convergence_semi_angle::typeof(1.0mrad) = 0.0mrad
    detector_array_size::Int = 0
    defocus::typeof(1.0μm) = 0.0μm
    fourier_space_sampling::typeof(1.0mrad) = 0.0mrad
    real_space_sampling::typeof(1.0Å) = 0.0Å
    wavelength::typeof(1.0nm) = 0.0nm
    amplitude_sum::Float64 = 0
end
ProbeParams(p::PtychoParams) = ProbeParams(p.convergence_semi_angle, p.detector_array_size, p.defocus, p.fourier_space_sampling, p.real_space_sampling, p.wavelength, p.amplitude_sum)

@option mutable struct SweepParams
    sweep_parameter::String="rotation"
    sweep_mode::String="pct"
    sweep_range::Vector=collect(1.0:1.0:1.0)
    sweep_metric::String="std"
end

@option mutable struct IterParams
    iteration_start::Int = 1
    iteration_end::Int = 1
    method::String = "ePIE"
    alpha::Float32 = 0.01
    beta::Float32 = 0.01
    GPUs::Vector{Int} = Int[]
    plotting::Bool = false
    shuffle::Bool = true
    filename::String = ""
    object_name::String = ""
    probe_name::String = ""
    sweep::SweepParams = SweepParams()
end
# IterParams(p::PtychoParams) = IterParams(p.iteration_start, p.iteration_end, p.alpha, p.beta, p.gpu, p.shuffle, p.filename, p.object_name, p.probe_name)

function unitAsString(unitOfQuantity::Unitful.FreeUnits) 
    replace(repr(unitOfQuantity,context = Pair(:fancy_exponent,false)), " " => "*")
end

OT = Union{PtychoParams, ObjectParams, ProbeParams, IterParams, SweepParams}
Configurations.from_dict(::Type{T} where T<:OT, ::Type{T} where T<:Unitful.Quantity, x) = x[1] * uparse(x[2])
Configurations.to_dict(::Type{T} where T<:OT, x::T where T<:Unitful.Quantity) = [ustrip(x), unitAsString(unit(x))]

function params_from_toml(::Type{T}, toml_file::String) where T<:OT
    from_dict(T, TOML.parsefile(toml_file))
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

function make_grid(dₛ, θᵣ, n₁, n₂; offset=[zero(dₛ), zero(dₛ)])
    init_grid = [[(cos(θᵣ)j - sin(θᵣ)i)dₛ, (cos(θᵣ)i + sin(θᵣ)j)dₛ] for (i,j) in product(1:n₁, 1:n₂)]
    min_x = minimum(first, init_grid)
    min_y = minimum(last, init_grid)
    grid = map(init_grid) do p
        p .- [min_x, min_y] .+ offset
    end
    return grid 
end
make_grid(dₛ, θᵣ, n; kwargs...) = make_grid(dₛ, θᵣ, n, n; kwargs...)
make_grid(p::PtychoParams; kwargs...) = make_grid(p.step_size, p.rotation_angle, first(p.scan_array_size), last(p.scan_array_size); kwargs...)

function make_object(grid, N, Δx, Δy; data_type=ComplexF32)
    min_x = minimum(first, grid)
    min_y = minimum(last, grid)
    max_x = maximum(first, grid)
    max_y = maximum(last, grid)

    padding_x = 0.5(N+1) * Δx
    padding_y = 0.5(N+1) * Δy

    𝒪_min_x = min_x - padding_x
    𝒪_min_y = min_y - padding_y 
    𝒪_max_x = max_x + padding_x
    𝒪_max_y = max_y + padding_y 

    nx = length(𝒪_min_x:Δx:𝒪_max_x)
    ny = length(𝒪_min_y:Δy:𝒪_max_y)

    𝒪 = AxisArray(ones(data_type, nx,ny); x = (𝒪_min_x:Δx:𝒪_max_x), y = (𝒪_min_y:Δy:𝒪_max_y))
    ℴ = map(grid) do p
        x1 = p[1] - Δx*N/2
        x2 = p[1] + Δx*N/2
        y1 = p[2] - Δy*N/2
        y2 = p[2] + Δy*N/2
        view(𝒪, x1 .. x2, y1 .. y2)
    end
    return 𝒪, ℴ
end
make_object(grid, N, Δx; data_type=ComplexF32) = make_object(grid, N, Δx, Δx; data_type=data_type)
make_object(op::ObjectParams; data_type=ComplexF32, kwargs...) = make_object(make_grid(op.step_size, op.rotation_angle, first(op.scan_array_size), last(op.scan_array_size); kwargs...), op.detector_array_size, op.real_space_sampling; data_type=data_type)
make_object(p::PtychoParams; data_type=ComplexF32, kwargs...) = make_object(ObjectParams(p); data_type=data_type, kwargs...)

function sum_sqrt_mean(dps)
    sum(sqrt.(mean(dps))) 
end

function make_probe(α, N, Δf, Δk, Δx, λ, mean_amplitude_sum; data_type=ComplexF32)
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
make_probe(pp::ProbeParams; kwargs...) = make_probe(pp.convergence_semi_angle, pp.detector_array_size, pp.defocus, pp.fourier_space_sampling, pp.real_space_sampling, pp.wavelength, pp.amplitude_sum; kwargs...)
make_probe(p::PtychoParams; kwargs...) = make_probe(ProbeParams(p); kwargs...)

function load_dps(filename, n₁, n₂)
    dps_mat = matread(filename)["dps"];
    dps = [dps_mat[:,:, i + (j-1)*n₁] for (i,j) in product(1:n₁, 1:n₂)]
    return dps
end
load_dps(filename, n) = load_dps(filename, n, n)

function make_amplitude(dps; data_type=Float32) 
    ThreadsX.map(x -> fftshift(sqrt.(x))|> Matrix{data_type}, dps)
end

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
    p1 = plot_amplitude(𝒲)
    p2 = plot_phase(𝒲; unwrap_phase=unwrap_phase)
    return plot(p1, p2, layout=(1,2))
end

function ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜; method="ePIE", ni=1, α=Float32(0.01), β=Float32(0.01), GPUs::Vector{Int}=Int[], plotting=false)
    ngpu = length(GPUs)
    for _ in 1:ni
        @time if ngpu == 0
            Threads.@threads for i in shuffle(eachindex(𝒜))
                ptycho_iteration!(ℴ[i], 𝒫, 𝒜[i]; method=method, α=α, β=β)
            end
        else 
            Threads.@threads for i in shuffle(eachindex(𝒜))
                CUDA.device!(GPUs[i % ngpu + 1])
                gpu_ptycho_iteration!(ℴ[i], 𝒫, 𝒜[i]; method=method, α=α, β=β)
            end
        end

        if plotting
            display(plot_wave(𝒫))
            display(plot_wave(𝒪))
        end
    end
    return nothing
end
ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, positions::UnitRange, kwargs...) = ptycho_reconstruction!(𝒪, ℴ[positions], 𝒫, 𝒜; kwargs...)

function ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, ip::IterParams)
    ni = length(range(ip.iteration_start, ip.iteration_end))
    ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜; method=ip.method, ni=ni, α=ip.alpha, β=ip.beta, GPUs=ip.GPUs, plotting=ip.plotting)
end


function save_object(filename, 𝒪; object_name="", object_params=ObjectParams(), data_type=ComplexF32)
    h5write(filename, join(filter(!isempty, ["/object", object_name]), "_"), convert(Matrix{data_type}, 𝒪))
    h5write(filename, join(filter(!isempty, ["/object", object_name, "params"]), "_"), to_toml(object_params))
end
save_object(𝒪, ip::IterParams; kwargs...) = save_object(ip.filename; object_name=ip.object_name, kwargs...)

function save_probe(filename, 𝒫; probe_name="", probe_params=ProbeParams(), data_type=ComplexF32)
    h5write(filename, join(filter(!isempty, ["/probe", probe_name]), "_"), convert(Matrix{data_type}, 𝒫))
    h5write(filename, join(filter(!isempty, ["/probe", probe_name, "params"]), "_"), to_toml(probe_params))
end
save_probe(𝒫, ip::IterParams; kwargs...) = save_probe(ip.filename; probe_name=ip.probe_name, kwargs...)

function save_result(filename, 𝒪, 𝒫; object_name="", probe_name="", ptycho_params=PtychoParams(), object_params=ObjectParams(ptycho_params), probe_params=ProbeParams(ptycho_params), data_type=ComplexF32)
    save_object(filename, 𝒪; object_name=object_name, object_params=object_params, data_type=data_type)
    save_probe(filename, 𝒫; probe_name=probe_name, probe_params=probe_params, data_type=data_type)
    h5write(filename, join(filter(!isempty, ["/ptycho", object_name, "params"]), "_"), to_toml(ptycho_params))
end
save_result(𝒪, 𝒫, ip::IterParams; kwargs...) = save_result(ip.filename, 𝒪, 𝒫; object_name=ip.object_name, probe_name=ip.probe_name, kwargs...)

function crop_center(im, w::Integer, h::Integer)
    m, n = size(im)
    l = floor((m/2 - w/2 + 1)) |> Int
    r = floor((m/2 + w/2)) |> Int
    b = floor((n/2 - h/2 + 1)) |> Int
    t = floor((n/2 + h/2)) |> Int
    im_out = im[l:r, b:t]
    return im_out
end
crop_center(im, l) = crop_center(im, l, l)

function cbed_center(cbed; threshold=0.1)
    bw_cbed = cbed .> (maximum(cbed) * threshold)
    cbed_indices = Tuple.(findall(x -> x==1, bw_cbed))
    Int.(round.(mean.((first.(cbed_indices), last.(cbed_indices)))))
end

function edge_distance(cbed, center)
    hcat(center .- (1, 1)...,  size(cbed) .- center...)
end

function shift_cbed(cbed; v=cbed_center(cbed))
    circshift(cbed, size(cbed)./2 .- v)
end

function align_cbeds(cbeds; threshold=0.1, crop=true)
    centers = ThreadsX.map(x -> cbed_center(x; threshold=threshold), cbeds)
    if crop
        all_distance = ThreadsX.map(edge_distance, cbeds, centers)
        crop_diameter = minimum(minimum.(all_distance)) * 2
        ThreadsX.map((x, y) -> crop_center(shift_cbed(x; v=y), crop_diameter), cbeds, centers)
    else
        ThreadsX.map((x, y) -> shift_cbed(x; v=y), cbeds, centers)
    end
end

function load_mib(filename::String; threshold=0.1, align=false, quadrant=1)
    data, _ = Medipix.load_mib(filename)
    if first(size(data)) == 512
        ranges = [[257:512, 257:512], [1:256, 257:512], [1:256, 1:256], [257:512, 1:256], [1:512, 1:512]]
        quadrant_range = ranges[quadrant]
        cbeds = map(x -> x[quadrant_range...], data)
    else
        cbeds = data
    end
    return align ? align_cbeds(cbeds, threshold=threshold) : cbeds
end

function parameter_sweep(𝒜, p₀::PtychoParams, ip₀::IterParams)
    # preserve original params
    p = p₀
    ip = ip₀

    parameter = ip.sweep.parameter 
    mode = ip.sweep.mode 
    range = ip.sweep.range 
    metric = ip.sweep.metric 
    if parameter ∉ ["rotation", "defocus", "step_size"]
        @error "$parameter sweep is not implemented. Possible parameters: rotation, defocus, and step_size"
    end
    
    if mode ∉ ["pct", "value"]
        @error "$mode mode is not implemented. Possible modes: pct and value"
    end

    if metric ∉ ["std", "max", "min", "mean"]
        @error "$metric is not one of the implemented metrics. Possible metrics: std, max, min, and mean"
    end

    sweep_result = map(range) do x
        if parameter == "rotation"
            δ = mode == "pct" ? p₀.rotation_angle * x : x
            p.rotation_angle = δ
        elseif parameter == "defocus"
            δ = mode == "pct" ? p₀.defocus * x : x
            p.defocus = δ
        elseif parameter == "step_size"
            δ = mode == "pct" ? p₀.step_size * x : x
            p.step_size = δ
        end

        ip.object_name = join(filter(!isempty, [parameter, mode, string(lpad(ustrip(x), 8, "0"))]), "_")
        ip.probe_name = join(filter(!isempty, [parameter, mode, string(lpad(ustrip(x), 8, "0"))]), "_")

        𝒪, ℴ = make_object(p)
        𝒫 = make_probe(p)
        ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, ip)

        if ip.filename != ""
            save_result(𝒪, 𝒫, ip; ptycho_params=p)
        end

        phase = angle.(𝒪)
        if parameter == "rotation"
            p.rotation_angle = 0°
            object_size = min(size(first(make_object(p)))...)
            selection_aperture = circular_aperture(object_size, object_size / 2 - 1)
            phase = std(selection_aperture .* crop_center(phase, object_size))
        end

        if metric == "std"
            result = std(phase)
        elseif metric == "max"
            result = maximum(phase)
        elseif metric == "min"
            result = minimum(phase)
        elseif metric == "mean"
            result = mean(phase)
        end

        return δ, result
    end

    if ip.filename != ""
        h5write(ip.filename, join("/result", parameter, mode, "sweep"), [ustrip(first.(sweep_result)) last.(sweep_result)])
    end
    return sweep_result
end

function filter_positions(center, positions, r)
    filter(x -> euclidean(center, x) < r, positions)
end

end
