using Revise
Revise.includet("tutorial-vumps-solutions/mps.jl")


function plotengising(n::Integer, D::Integer)
    pts = range(0.1, 2, length=n)
    βc = log(1+sqrt(2))/2

    Z = Array{Float64}([1. 0. ; 0 -1])

    temps = []
    values = []

    io = open("data.txt","w")

    for x = pts
        print(n)

        β = x*βc
        M, M2, M3 = classicalisingmpo(β)
        A = randn(D, 2, D) + im*randn(D, 2, D)
        λ, AL, C, AR, FL, FR = vumps(A, M; verbose = false, tol = 1e-10)


        @tensor AAC[α,s1,s2,β] := AL[α,s1,α']*C[α',β']*AR[β',s2,β]

        @tensor Z2 = scalar(FL[α,c,β]*AAC[β,s1,s2,β']*M[c,t1,d,s1]*M[d,t2,c',s2]*FR[β',c',α']*conj(AAC[α,t1,t2,α']))
        @tensor energy = scalar(FL[α,c,β]*AAC[β,s1,s2,β']*M2[c,t1,t2,c',s2,s1]*FR[β',c',α']*conj(AAC[α,t1,t2,α']) / Z2)
        @tensor magn = scalar(FL[α,c,β]*AAC[β,s1,s2,β']*M3[c,t1,t2,c',s2,s1]*FR[β',c',α']*conj(AAC[α,t1,t2,α']) / Z2)

        #WW = magmpo()
        #Z = [1. 0. ; 0. -1.]

        #@tensor magn = scalar(FL[α,c,β]*AAC[β,s1,s2,β']*WW[c,t1,t2,c',s2,s1]*FR[β',c',α']*conj(AAC[α,t1,t2,α']))

        #@tensor AC[α, s, β]:= AL[α,s,β']*C[β',β]
        #@tensor magn = scalar(AC[α, s1, β]*conj(AC[α,s2,β])*Z[s2,s1])

        println(io, string(x), '\t', string(real(energy)), '\t', string(real(magn)))

        temps = push!(temps, x)
        values = push!(values, real(energy))
    end
    close(io)
    return temps, values
end

function statmechmpo(β, h, D)
    M = zeros(D,D,D,D)
    for i = 1:D
        M[i,i,i,i] = 1
    end
    X = zeros(D,D)
    for j = 1:D, i = 1:D
        X[i,j] = exp(-β*h(i,j))
    end
    Xsq = sqrt(X)
    @tensor M1[a,b,c,d] := M[a',b',c',d']*Xsq[c',c]*Xsq[d',d]*Xsq[a,a']*Xsq[b,b']

    # For computing energy: M2 is a tensor across 2 nearest neighbor sites in the lattice, whose
    # expectation value in the converged fixed point of the transfer matrix represents the energy
    Y = zeros(D,D)
    for j = 1:D, i = 1:D
        Y[i,j] = h(i,j)*exp(-β*h(i,j))
    end
    @tensor M2[a,b1,b2,c,d2,d1] := M[a',b1',c1,d1']*Xsq[a,a']*Xsq[b1,b1']*Xsq[d1',d1]* Y[c1,c2]*
                                    M[c2,b2',c',d2']*Xsq[b2,b2']*Xsq[d2',d2]*Xsq[c',c]

    Z0 = [1. 0. ;0. -1.]
    Z = zeros(D,D)
    for j = 1:D, i = 1:D
        Z[i,j] = Z0[i,j]*exp(-β*h(i,j))
    end
    @tensor M3[a,b1,b2,c,d2,d1] := M[a',b1',c1,d1']*Xsq[a,a']*Xsq[b1,b1']*Xsq[d1',d1]* Z[c1,c2]*
                                    M[c2,b2',c',d2']*Xsq[b2,b2']*Xsq[d2',d2]*Xsq[c',c]

    return M1, M2, M3
end

function classicalisingmpo(β; J = 1.0, h = 0.)
    statmechmpo(β, (s1,s2)->-J*(-1)^(s1!=s2) - h/2*(s1==1 + s2==1), 2)
end


function bloch(a,D)
    t = zeros(ComplexF64,D,2,2)
    id = [1. 0.; 0. 1.]
    X = [0. 1.;1. 0.]
    Y = [0. im ;-im 0.]
    Z = [1. 0. ; 0. -1.]
    ρ = 0.5*(id + a[1]*X + a[2]*Y + a[3]*Z)
    t[1,:,:]=ρ
    print(ρ)
    @tensor ρt[a,b,c,d,s1,s2] := t[a,t1,s2]*t[b,t2,t1]*t[c,t3,t2]*t[d,s1,t3]
    @tensor rv[a,b,c,d] := ρt[a,b,c,d,s,s]

    return ρt,rv
end

function vumpsbloch(a,D)
    ρ, M = bloch(a,D)
    A = randn(1, D, 1) + im*randn(1, D, 1)

    λ, AL, C, AR, FL, FR = vumps(A, M; verbose = false, tol = 1e-10)

    @tensor AC[a,s,b] := AL[a,s,c]*C[c,b]
    @tensor ρA[t1,t2] := FL[a',c1,a]*AC[a,s1,b]*FR[b,c2,b']*conj(AC[a',s2,b'])*ρ[c1,s2,c2,s1,t1,t2]

    return ρA
end
