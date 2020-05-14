using NestedSamplers.Bounds: Ellipsoid, MultiEllipsoid, fit, scale!, span, decompose, volume, volume_prefactor

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
        @test axs ≈ span(ell) ≈ diagm(0 => fill(scale, N))
    end

    @testset "Scaling" begin
        scale = 1.5
        center = zeros(N)
        A = diagm(0 => rand(N))
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

        A = diagm(0 => rand(N))
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

    @testset "Ellipsoid Sample" begin
        nsamples = 1000
        volfrac = 0.5
        ell = random_ellipsoid(N)
        ell2 = deepcopy(ell)
        scale!(ell2, volfrac)

        # expected number of points that will fall within inner ellipsoid
        expect = volfrac * nsamples
        σ = sqrt((1 - volfrac) * expect)

        # sample randomly
        ninner = 0
        for i in 1:nsamples
            x = rand(ell)
            @test x ∈ ell
            ninner += Int(x ∈ ell2)
        end

        @test ninner ≈ expect atol = 5σ
    end

    @testset "Bounding" begin
        ell_gen = random_ellipsoid(N)
        x = rand(ell_gen, 100)
        ell = fit(Ellipsoid, x)
        @test all(x[:, i] ∈ ell for i in axes(x, 2))
    end

    @testset "Bounding Robust" begin
        ell_gen = random_ellipsoid(N)
        x = rand(ell_gen, N)
        for npoints in 1:N
            ell = fit(Ellipsoid, x[:, 1:npoints], pointvol = volume(ell_gen) / npoints)

            @test volume(ell) ≈ volume(ell_gen) rtol = 1e-5
            @test all(x[:, i] ∈ ell for i in 1:npoints)
        end
    end

    @testset "ME Scaling" begin
        scale = 1.5
        ells = [random_ellipsoid(N) for _ in 1:6]
        me = MultiEllipsoid(ells)
        v1 = volume(me)
        scale!(me, scale)

        @test volume(me) ≈ scale * v1 rtol = 1e-6
    end

    @testset "ME Sample" begin
        nsamples = 1000
        volfrac = 0.5
    
        ells = [random_ellipsoid(N) for _ in 1:6]
        me = MultiEllipsoid(ells)
        me2 = deepcopy(me)
        scale!(me2, volfrac)

        # expected number of points that will fall within inner ellipsoid
        expect = volfrac * nsamples
        σ = sqrt((1 - volfrac) * expect)

        # sample randomly
        ninner = 0
        for i in 1:nsamples
            x = rand(me)
            @test x ∈ me
            ninner += Int(x ∈ me2)
        end

        @test ninner ≈ expect atol = 5σ
    end

    @testset "ME Bounding" begin
        ell_gens = [random_ellipsoid(N) for _ in 1:6]
        me_gen = MultiEllipsoid(ell_gens)
        x = rand(me_gen, 100)
        ell = fit(MultiEllipsoid, x)
        @test all(x[:, i] ∈ ell for i in axes(x, 2))
    end

end # testset
