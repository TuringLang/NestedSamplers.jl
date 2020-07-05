using NestedSamplers.Bounds: Ellipsoid, MultiEllipsoid, fit, scale!, decompose, volume, volume_prefactor

const NMAX = 20

@testset "ndims=$N" for N in 1:NMAX  
    @testset "Spheres" begin 
        scale = 5
        center = 2scale .* ones(N)
        A = diagm(0 => ones(N) ./ scale^2)
        ell = Ellipsoid(center, A)
        @test volume(ell) ≈ volume_prefactor(N) * scale^N
        axs, axlens = decompose(ell)
        @test axlens ≈ fill(scale, N)
        @test axs ≈ Bounds.axes(ell) ≈ diagm(0 => fill(scale, N))
    end

    @testset "Scaling" begin
        scale = 1.5
        center = zeros(N)
        A = diagm(0 => rand(rng, N))
        ell = Ellipsoid(center, A)

        ell2 = Ellipsoid(center, A ./ scale^2)

        scale!(ell, scale^N)

        @test volume(ell) ≈ volume(ell2)
        @test ell.A ≈ ell2.A
        @test all(decompose(ell) .≈ decompose(ell2))
    end

    @testset "Contains" begin
        E = 1e-7
        ell = Ellipsoid(N)

        # Point just outside unit n-Spheres
        pt = (1 / √N + E) .* ones(N)
        @test pt ∉ ell

        # point just inside
        pt = (1 / √N - E) .* ones(N)
        @test pt ∈ ell

        A = diagm(0 => rand(rng, N))
        ell = Ellipsoid(zeros(N), A)

        for i in 1:N
            axlen = 1 / sqrt(A[i, i])
            pt = zeros(N)
            pt[i] = axlen + E
            @test pt ∉ ell
            pt[i] = axlen - E
            @test pt ∈ ell
        end
    end
end # testset
