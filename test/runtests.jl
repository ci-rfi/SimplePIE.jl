using SimplePIE
using Unitful: Å, nm, μm, °, kV, mrad
using Unitful
using Test

@testset "SimplePIE.jl" begin
    # Write your tests here.
    project = "test_project"
    session = "test_session"
    datadir = "test_dir"
    datafile = "test_data.h5"
    timestamp = "2022-01-01T01:01:01.001"

    N = [256, 256]
    n = [127, 127]
    λ = wavelength(300kV)
    α = 1.0340mrad
    D = 125
    Δk = 2α / D
    θ = N * Δk
    θᵣ = -126°
    dₛ = 31.25Å
    Δx, Δy = uconvert.(Å, λ./θ)
    Δf = -13μm
    rₚ = probe_radius(α, Δf)
    sₚ = probe_area(α, Δf)
    overlap, overlap_ratio = probe_overlap(rₚ, dₛ; ratio=true)
    scaling_factor = 47317.77435855447

    data_params = DataParams(project, session, datadir, datafile, timestamp, N, n, λ, α, Δk, θ, θᵣ, dₛ, Δx, Δf, rₚ, sₚ, overlap, overlap_ratio, scaling_factor)
    object_params = ObjectParams(dₛ, θᵣ, n, N, Δx)
    probe_params = ProbeParams(α, N, Δf, Δk, Δx, λ, scaling_factor) 

    data_params_from_toml = from_toml(DataParams, "data_params.toml")
    object_params_from_toml = from_toml(ObjectParams, "object_params.toml")
    probe_params_from_toml = from_toml(ProbeParams, "probe_params.toml")

    @test object_params == object_params_from_toml
    @test probe_params == probe_params_from_toml

    object_params_from_data_params = ObjectParams(data_params)
    probe_params_from_data_params = ProbeParams(data_params)

    @test object_params == object_params_from_data_params
    @test probe_params == probe_params_from_data_params
end
