using SimplePIE
using Unitful: Å, nm, μm, °, kV, mrad
using Unitful
using Test

@testset "SimplePIE.jl" begin
    # Write your tests here.
    N = 256
    n = 127
    λ = wavelength(300kV)
    α = 1.0340mrad
    D = 125
    Δk = 2α / D
    θ = N * Δk
    θᵣ = -126°
    dₛ = 31.25Å
    Δx = uconvert(Å, λ/θ)
    Δf = -13μm

    ptycho_params = PtychoParams(N, n, λ, α, Δk, θ, θᵣ, dₛ, Δx, Δf)
    object_params = ObjectParams(dₛ, θᵣ, n, N, Δx)
    probe_params = ProbeParams(α, N, Δf, Δk, Δx, λ)

    ptycho_params_from_toml = params_from_toml(PtychoParams, "ptycho_params.toml")
    object_params_from_toml = params_from_toml(ObjectParams, "object_params.toml")
    probe_params_from_toml = params_from_toml(ProbeParams, "probe_params.toml")

    @test object_params == object_params_from_toml
    @test probe_params == probe_params_from_toml

    object_params_from_ptycho_params = ObjectParams(ptycho_params)
    probe_params_from_ptycho_params = ProbeParams(ptycho_params)

    @test object_params == object_params_from_ptycho_params
    @test probe_params == probe_params_from_ptycho_params
end
