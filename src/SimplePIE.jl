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

export DataParams
export ObjectParams
export ProbeParams
export ReconParams
export SweepParams
export from_toml

export wavelength
export circular_aperture
export make_grid
export make_object
export cbed_power
export make_probe
export probe_radius
export probe_area
export probe_overlap
export load_cbeds
export load_mat
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
export load_object
export load_probe
export load_result
export crop_center
export cbed_center
export edge_distance
export shift_cbed
export align_cbeds
export rotation_angle_sweep
export stepsize_sweep
export defocus_sweep
export parameter_sweep
export positions_in_roi
export linear_positions

@option mutable struct DataParams
    project::String = "default_project"
    session::String = "default_session"
    datadir::String = ""
    datafile::String = "ptycho_data.h5"
    timestamp::String = "2022-01-01T01:01:01.001"
    detector_array_size::Vector{Int} = [0, 0]
    scan_array_size::Vector{Int} = [0, 0]
    wavelength::typeof(1.0nm) = 0.0nm
    convergence_semi_angle::typeof(1.0mrad) = 0.0mrad
    fourier_space_sampling::typeof(1.0mrad) = 0.0mrad
    maximum_angle::typeof([1.0mrad, 1.0mrad]) = [0.0mrad, 0.0mrad]
    rotation_angle::typeof(1.0°) = 0.0°
    step_size::typeof(1.0Å) = 0.0Å
    real_space_sampling::typeof(1.0Å) = 0.0Å
    defocus::typeof(1.0μm) = 0.0μm
    probe_radius::typeof(1.0nm) = 0.0nm
    probe_area::typeof(1.0nm^2) = 0.0nm^2
    overlap::typeof(1.0nm^2) = 0.0nm^2
    overlap_ratio::typeof(1.0u"percent") = 0.0u"percent"
    probe_power::Float64 = 1.0
end

@option mutable struct ObjectParams
    step_size::typeof(1.0Å) = 0.0Å
    rotation_angle::typeof(1.0°) = 0.0°
    scan_array_size::Vector{Int} = [0, 0]
    detector_array_size::Vector{Int} = [0, 0]
    real_space_sampling::typeof(1.0Å) = 0.0Å
end
ObjectParams(dp::DataParams) = ObjectParams(dp.step_size, dp.rotation_angle, dp.scan_array_size, dp.detector_array_size, dp.real_space_sampling)

@option mutable struct ProbeParams
    convergence_semi_angle::typeof(1.0mrad) = 0.0mrad
    detector_array_size::Vector{Int} = [0, 0]
    defocus::typeof(1.0μm) = 0.0μm
    fourier_space_sampling::typeof(1.0mrad) = 0.0mrad
    real_space_sampling::typeof(1.0Å) = 0.0Å
    wavelength::typeof(1.0nm) = 0.0nm
    probe_power::Float64 = 1.0
end
ProbeParams(dp::DataParams) = ProbeParams(dp.convergence_semi_angle, dp.detector_array_size, dp.defocus, dp.fourier_space_sampling, dp.real_space_sampling, dp.wavelength, dp.probe_power)


@option mutable struct SweepParams
    parameter::String="rotation"
    mode::String="pct"
    range::Vector=collect(1.0:1.0:1.0)
    metric::String="std"
end

@option mutable struct ReconParams
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
end

function unitAsString(unitOfQuantity::Unitful.FreeUnits) 
    replace(repr(unitOfQuantity,context = Pair(:fancy_exponent,false)), " " => "*")
end

OT = Union{DataParams, ObjectParams, ProbeParams, ReconParams, SweepParams}
Configurations.from_dict(::Type{T} where T<:OT, ::Type{T} where T<:Unitful.Quantity, x) = x[1] * uparse(x[2] == "%" ? "percent" : x[2])
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

function circular_aperture(n, r; shift=CartesianIndex(0, 0), σ=0)
    data = Matrix{Bool}(undef, first(n), last(n))
    if min(n...) <= 2r 
        @warn("Aperature area exceeds the field of view even if centered.") 
    end
    origin =  CartesianIndex(ceil.(Int, size(data) ./ 2)...) + shift
    for ind in CartesianIndices(data)
        data[ind] = hypot(Tuple(ind - origin)...) <= r ? true : false
    end
    aperture = imfilter(data, Kernel.gaussian(σ))
    return aperture
end

function make_grid(dₛ, θᵣ, n; offset=[zero(dₛ), zero(dₛ)])
    n₁ = first(n)
    n₂ = last(n)
    init_grid = [[(cos(θᵣ)j - sin(θᵣ)i)dₛ, (cos(θᵣ)i + sin(θᵣ)j)dₛ] for (i,j) in product(1:n₁, 1:n₂)]
    min_x = minimum(first, init_grid)
    min_y = minimum(last, init_grid)
    grid = map(init_grid) do p
        p .- [min_x, min_y] .+ offset
    end
    return grid 
end
make_grid(dp::DataParams; kwargs...) = make_grid(dp.step_size, dp.rotation_angle, dp.scan_array_size; kwargs...)

function make_object(grid, N, Δx; data_type=ComplexF32)
    N₁ = first(N)
    N₂ = last(N)
    Δy = Δx

    min_x = minimum(first, grid)
    min_y = minimum(last, grid)
    max_x = maximum(first, grid)
    max_y = maximum(last, grid)

    padding_x = 0.5(N₁+1) * Δx
    padding_y = 0.5(N₂+1) * Δy

    𝒪_min_x = min_x - padding_x
    𝒪_min_y = min_y - padding_y 
    𝒪_max_x = max_x + padding_x
    𝒪_max_y = max_y + padding_y 

    nx = length(𝒪_min_x:Δx:𝒪_max_x)
    ny = length(𝒪_min_y:Δy:𝒪_max_y)

    𝒪 = AxisArray(ones(data_type, nx, ny); x = (𝒪_min_x:Δx:𝒪_max_x), y = (𝒪_min_y:Δy:𝒪_max_y))
    ℴ = map(grid) do p
        x1 = p[1] - Δx*N₁/2
        x2 = p[1] + Δx*N₁/2
        y1 = p[2] - Δy*N₂/2
        y2 = p[2] + Δy*N₂/2
        view(𝒪, x1 .. x2, y1 .. y2)
    end
    return 𝒪, ℴ
end
make_object(op::ObjectParams; data_type=ComplexF32, kwargs...) = make_object(make_grid(op.step_size, op.rotation_angle, op.scan_array_size; kwargs...), op.detector_array_size, op.real_space_sampling; data_type=data_type)
make_object(dp::DataParams; data_type=ComplexF32, kwargs...) = make_object(ObjectParams(dp); data_type=data_type, kwargs...)

function cbed_power(cbed_ref)
    sum(cbed_ref)
end

function make_probe(α, N, Δf, Δk, Δx, λ, probe_power; data_type=ComplexF32)
    N₁ = first(N)
    N₂ = last(N)
    Δy = Δx

    K = [Δk * [i,j] for (i,j) in product(-N₁/2:N₁/2-1, -N₂/2:N₂/2-1)]
    ω = map(x -> x[1] + x[2]im, K)
    ωᵢ = map(x -> x[1] - x[2]im, K)
    χ = (ω .* ωᵢ) * Δf / 2 |> real |> x -> uconvert.(nm, x)
    aberration = -2π/λ * χ 
    aperture = circular_aperture(N, Int(round(α/Δk)); σ=1) 
    𝒟 = cis.(aberration) .* aperture 
    𝒟 = 𝒟 / sum(abs.(𝒟))
    𝒫_array = fftshift(ifft(ifftshift(𝒟))) 

    𝒻ₜ = √prod(size(𝒟))
    𝒻ₚ = √probe_power
    𝒫_array = 𝒫_array * 𝒻ₚ / 𝒻ₜ / √sum(abs.(𝒫_array).^2) |> Matrix{data_type}

    𝒫_min_x = -0.5(N₁+1) * Δx
    𝒫_max_x = 0.5(N₁-2) * Δx
    𝒫_min_y = -0.5(N₂+1) * Δy
    𝒫_max_y = 0.5(N₂-2) * Δy
    𝒫 = AxisArray(𝒫_array; x = (𝒫_min_x:Δx:𝒫_max_x), y = (𝒫_min_y:Δy:𝒫_max_y))
    return 𝒫
end
make_probe(pp::ProbeParams; data_type=ComplexF32) = make_probe(pp.convergence_semi_angle, pp.detector_array_size, pp.defocus, pp.fourier_space_sampling, pp.real_space_sampling, pp.wavelength, pp.probe_power; data_type=data_type)
make_probe(dp::DataParams; data_type=ComplexF32) = make_probe(ProbeParams(dp); data_type=data_type)

function probe_radius(α, Δf)
    return uconvert(nm, abs(tan(α) * Δf))
end

function probe_area(α, Δf)
    rₚ = probe_radius(α, Δf)
    return π * rₚ^2
end

function probe_overlap(rₚ, dₛ; ratio=false)
    overlap_area = uconvert(nm^2, 2*rₚ^2*acos(dₛ/2rₚ) - 0.5*dₛ*√(4rₚ^2 - dₛ^2))
    circle_area = π * rₚ^2
    overlap_ratio = upreferred(overlap_area / circle_area) * 100u"percent"
    return ratio ? (overlap_area, overlap_ratio) : overlap_area
end

function load_cbeds(f, filename::String; quadrant=0, align=false, threshold=0.1, crop=true)
    cbeds = f(filename)
    if quadrant ∈ [1, 2, 3, 4]
        l₁, l₂ = Int.(round.(size(first(cbeds)) ./ 2)) 
        ranges = [[l₁+1:2l₁, l₂+1:2l₂], [1:l₁, l₂+1:2l₂], [1:l₁, 1:l₂], [l₁+1:2l₁, 1:l₂]]
        cbeds = map(x -> x[ranges[quadrant]...], cbeds)
    end
    return align ? align_cbeds(cbeds; threshold=threshold, crop=crop) : cbeds
end
load_cbeds(filename; kwargs...) = load_cbeds(x->load_mat(x), filename; kwargs...)

# function load_h5(filename)
#     cbeds = h5read(filename, "dps")
#     return cbeds
# end

function load_mat(filename)
    mat = matread(filename)["dps"]
    cbeds = [mat[:,:,i] for i in 1:last(size(mat))]
    return cbeds
end

function load_mib(filename)
    cbeds, _ = Medipix.load_mib(filename)
    return cbeds
end

function make_amplitude(cbeds; data_type=Float32) 
    ThreadsX.map(x -> ifftshift(sqrt.(x))|> Matrix{data_type}, cbeds)
end

function update!(q, a, Δψ; method="ePIE", α=0.2) 
    if iszero(α) 
        return nothing
    end
    α = convert(eltype(real(q)), α)

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
        @error "$method is not a supported update method."
    end

    q[:] = q + w .* Δψ
    return nothing
end

function ptycho_iteration!(𝒪, 𝒫, 𝒜; method="ePIE", α=0.2, β=0.2, probe_power=1.0)
    𝒻ₜ = convert(eltype(real(𝒫)), √prod(size(𝒫)))
    𝒻ₚ = convert(eltype(real(𝒫)), √probe_power)

    ψ₁ = 𝒪 .* 𝒫
    𝒟 = 𝒜 / 𝒻ₜ .* sign.(fft(ifftshift(ψ₁)))
    ψ₂ = fftshift(ifft(𝒟)) * 𝒻ₜ
    Δψ = ψ₂ - ψ₁

    𝒫[:] = 𝒫 * 𝒻ₚ / 𝒻ₜ / √sum(abs.(𝒫).^2)
    update!(𝒪, 𝒫, Δψ; method=method, α=α)
    update!(𝒫, 𝒪, Δψ; method=method, α=β)
    return nothing
end

function gpu_ptycho_iteration!(𝒪_cpu, 𝒫_cpu, 𝒜_cpu; method="ePIE", α::Float32=Float32(0.2), β::Float32=Float32(0.2), probe_power=1.0)
    𝒪 = CuArray(copy(𝒪_cpu.data))
    𝒫 = CuArray(copy(𝒫_cpu.data))
    𝒜 = CuArray(𝒜_cpu)
    ptycho_iteration!(𝒪, 𝒫, 𝒜; method=method, α=α, β=β, probe_power=probe_power)
    copyto!(𝒪_cpu, Array(𝒪))
    copyto!(𝒫_cpu, Array(𝒫))
    return nothing
end

function plot_amplitude(𝒲; with_unit=true, kwargs...)
    amplitude = abs.(𝒲)
    if with_unit
        return heatmap(amplitude; aspect_ratio=:equal, xlim=(1, size(amplitude, 1)), ylim=(1, size(amplitude, 2)), xrotation=-20, xformatter= x -> round(typeof(1nm), 𝒲.axes[1][Int(x)]), yformatter= y -> round(typeof(1nm), 𝒲.axes[2][Int(y)]), kwargs...)
    else
        return heatmap(amplitude; aspect_ratio=:equal)
    end
end

function plot_phase(𝒲; unwrap_phase=false, with_unit=true, kwargs...)
    phase = unwrap_phase ? unwrap(angle.(𝒲); dims=1:2) : angle.(𝒲)
    if with_unit
        return heatmap(phase; aspect_ratio=:equal, xlim=(1, size(phase, 1)), ylim=(1, size(phase, 2)), xrotation=-20, xformatter= x -> round(typeof(1nm), 𝒲.axes[1][Int(x)]), yformatter= y -> round(typeof(1nm), 𝒲.axes[2][Int(y)]), kwargs...)
    else
        return heatmap(phase; aspect_ratio=:equal)
    end
end

function plot_wave(𝒲; unwrap_phase=false, with_unit=true, kwargs...)
    p1 = plot_amplitude(𝒲; with_unit=with_unit, kwargs...)
    p2 = plot_phase(𝒲; unwrap_phase=unwrap_phase, with_unit=with_unit, kwargs...)
    return plot(p1, p2, layout=(1,2))
end

function ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜; method="ePIE", ni=1, α=Float32(0.01), β=Float32(0.01), probe_power=1.0, GPUs::Vector{Int}=Int[], plotting=false)
    ngpu = length(GPUs)
    for _ in 1:ni
        @time if ngpu == 0
            Threads.@threads for i in shuffle(eachindex(𝒜))
                ptycho_iteration!(ℴ[i], 𝒫, 𝒜[i]; method=method, α=α, β=β, probe_power=probe_power)
            end
        else 
            Threads.@threads for i in shuffle(eachindex(𝒜))
                CUDA.device!(GPUs[i % ngpu + 1])
                gpu_ptycho_iteration!(ℴ[i], 𝒫, 𝒜[i]; method=method, α=α, β=β, probe_power=probe_power)
            end
        end

        if plotting
            display(plot_wave(𝒫))
            display(plot_wave(𝒪))
        end
    end
    return nothing
end

function ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, dp::DataParams, rp::ReconParams)
    ni = length(range(rp.iteration_start, rp.iteration_end))
    ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜; method=rp.method, ni=ni, α=rp.alpha, β=rp.beta, probe_power=dp.probe_power, GPUs=rp.GPUs, plotting=rp.plotting)
end

function save_object(filename, 𝒪; object_name="", object_params=ObjectParams(), data_type=ComplexF32)
    h5write(filename, join(filter(!isempty, ["/object", object_name]), "_"), convert(Matrix{data_type}, 𝒪))
    h5write(filename, join(filter(!isempty, ["/object_params", object_name]), "_"), to_toml(object_params))
end
save_object(𝒪, rp::ReconParams; kwargs...) = save_object(rp.filename, 𝒪; object_name=rp.object_name, kwargs...)

function save_probe(filename, 𝒫; probe_name="", probe_params=ProbeParams(), data_type=ComplexF32)
    h5write(filename, join(filter(!isempty, ["/probe", probe_name]), "_"), convert(Matrix{data_type}, 𝒫))
    h5write(filename, join(filter(!isempty, ["/probe_params", probe_name]), "_"), to_toml(probe_params))
end
save_probe(𝒫, rp::ReconParams; kwargs...) = save_probe(rp.filename, 𝒫; probe_name=rp.probe_name, kwargs...)

function save_result(filename, 𝒪, 𝒫; object_name="", probe_name=object_name, data_params=DataParams(), recon_params=ReconParams(), object_params=ObjectParams(data_params), probe_params=ProbeParams(data_params), data_type=ComplexF32)
    save_object(filename, 𝒪; object_name=object_name, object_params=object_params, data_type=data_type)
    save_probe(filename, 𝒫; probe_name=probe_name, probe_params=probe_params, data_type=data_type)
    h5write(filename, join(filter(!isempty, ["/data_params", object_name]), "_"), to_toml(data_params))
    h5write(filename, join(filter(!isempty, ["/recon_params", object_name]), "_"), to_toml(recon_params))
end
save_result(𝒪, 𝒫, rp::ReconParams; kwargs...) = save_result(rp.filename, 𝒪, 𝒫; object_name=rp.object_name, probe_name=rp.probe_name, recon_params=rp, kwargs...)
save_result(𝒪, 𝒫, dp::DataParams, rp::ReconParams; kwargs...) = save_result(rp.filename, 𝒪, 𝒫; object_name=rp.object_name, probe_name=rp.probe_name, data_params=dp, recon_params=rp, kwargs...)

function load_object(filename; object_name="", object_params=ObjectParams(), data_type=ComplexF32)
    if object_params == ObjectParams()
        object_params = from_dict(ObjectParams, TOML.parse(h5read(filename, join(filter(!isempty, ["/object_params", object_name]), "_"))))
    end
    𝒪, ℴ = make_object(object_params; data_type=data_type)
    𝒪[:] = h5read(filename, join(filter(!isempty, ["/object", object_name]), "_"))
    return 𝒪, ℴ
end
load_object(rp::ReconParams; kwargs...) = load_object(rp.filename; object_name=rp.object_name, kwargs...)

function load_probe(filename; probe_name="", probe_params=ProbeParams(), data_type=ComplexF32)
    if probe_params == ProbeParams()
        probe_params = from_dict(ProbeParams, TOML.parse(h5read(filename, join(filter(!isempty, ["/probe_params", probe_name]), "_"))))
    end
    𝒫 = make_probe(probe_params; data_type=data_type)
    𝒫[:] = h5read(filename, join(filter(!isempty, ["/probe", probe_name]), "_"))
    return 𝒫
end
load_probe(rp::ReconParams; kwargs...) = load_probe(rp.filename; probe_name=rp.probe_name, kwargs...)

function load_result(filename; object_name="", probe_name=object_name, data_type=ComplexF32)
    data_params = from_dict(DataParams, TOML.parse(h5read(filename, join(filter(!isempty, ["/data_params", object_name]), "_"))))
    recon_params = from_dict(ReconParams, TOML.parse(h5read(filename, join(filter(!isempty, ["/recon_params", object_name]), "_"))))
    𝒪, ℴ = load_object(filename; object_name=object_name, object_params=ObjectParams(data_params), data_type=data_type)
    𝒫 = load_probe(filename; probe_name=probe_name, probe_params=ProbeParams(data_params), data_type=data_type)
    return data_params, recon_params, 𝒪, ℴ, 𝒫
end
load_result(rp::ReconParams; kwargs...) = load_result(rp.filename; object_name=rp.object_name, probe_name=rp.probe_name, recon_params=rp, kwargs...)
load_result(dp::DataParams, rp::ReconParams; kwargs...) = load_result(rp.filename; object_name=rp.object_name, probe_name=rp.probe_name, data_params=dp, recon_params=rp, kwargs...)

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

function parameter_sweep(𝒜, dp₀::DataParams, rp₀::ReconParams, sp₀::SweepParams)
    # preserve original params
    dp = from_dict(DataParams, to_dict(dp₀))
    rp = from_dict(ReconParams, to_dict(rp₀))

    if sp₀.parameter ∉ ["rotation", "defocus", "step_size"]
        @error "$(sp₀.parameter) sweep is not implemented. Possible parameters: rotation, defocus, and step_size"
    end
    
    if sp₀.mode ∉ ["pct", "value"]
        @error "$(sp₀.mode) mode is not implemented. Possible modes: pct and value"
    end

    if sp₀.metric ∉ ["std", "max", "min", "mean"]
        @error "$(sp₀.metric) is not one of the implemented metrics. Possible metrics: std, max, min, and mean"
    end

    sweep_result = map(sp₀.range) do x
        if sp₀.parameter == "rotation"
            δ = sp₀.mode == "pct" ? dp₀.rotation_angle * x : x
            dp.rotation_angle = δ
        elseif sp₀.parameter == "defocus"
            δ = sp₀.mode == "pct" ? dp₀.defocus * x : x
            dp.defocus = δ
        elseif sp₀.parameter == "step_size"
            δ = sp₀.mode == "pct" ? dp₀.step_size * x : x
            dp.step_size = δ
        end

        rp.object_name = join(filter(!isempty, [sp₀.parameter, sp₀.mode, string(round(ustrip(x), sigdigits=8))]), "_")
        rp.probe_name = join(filter(!isempty, [sp₀.parameter, sp₀.mode, string(round(ustrip(x), sigdigits=8))]), "_")

        𝒪, ℴ = make_object(dp)
        𝒫 = make_probe(dp)
        ptycho_reconstruction!(𝒪, ℴ, 𝒫, 𝒜, dp, rp)

        if rp.filename != ""
            save_result(𝒪, 𝒫, dp, rp)
        end

        phase = angle.(𝒪)
        if sp₀.parameter == "rotation"
            dp.rotation_angle = 0°
            object_size = min(size(first(make_object(dp)))...)
            selection_aperture = circular_aperture(object_size, object_size / 2 - 1)
            phase = selection_aperture .* crop_center(phase, object_size)
        end

        if sp₀.metric == "std"
            result = std(phase)
        elseif sp₀.metric == "max"
            result = maximum(phase)
        elseif sp₀.metric == "min"
            result = minimum(phase)
        elseif sp₀.metric == "mean"
            result = mean(phase)
        end

        return δ, result
    end

    if rp.filename != ""
        h5write(rp.filename, join(["sweep_result", sp₀.parameter, sp₀.mode], "_"), [ustrip(first.(sweep_result)) last.(sweep_result)])
        h5write(rp.filename, join(["sweep_params", sp₀.parameter, sp₀.mode], "_"), to_toml(sp₀))
    end
    return sweep_result
end

function positions_in_roi(center, positions, r; roi_shape="circle")
    if roi_shape == "circle"
        filter(x -> euclidean(center, x) < r, positions)
    elseif roi_shape == "square"
        filter(x -> all(euclidean.(center, x) .< r/2), positions)
    else
        @warn "\"$roi_shape\" is not a supported roi shape, default to \"circle\"."
        positions_in_roi(center, positions, r; roi_shape="circle")
    end
end

function linear_positions(grid, positions)
    if eltype(first(positions)) == Int
        map(x -> LinearIndices(grid)[x...], positions)
    else
        map(x -> LinearIndices(grid)[x], findall(!iszero, map(x -> x ∈ positions, grid)))
    end
end
linear_positions(dp::DataParams, positions; kwargs...) = linear_positions(make_grid(dp; kwargs...), positions)

end
