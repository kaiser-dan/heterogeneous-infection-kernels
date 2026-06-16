using CairoMakie, LogExpFunctions, ColorSchemes
inch = 96; pt = 4÷3

# q(β, p, r, j) = logistic(logit(β / p) + r * (j - 1))

# function c(β, p, r, ν)
#     return p * (1 - prod(1 - q(β, p, r, j) for j in 1:ν))
# end

# fig = Figure(size=(6inch, 2.5inch), fontsize=10pt)
# νs = 0:25

# # fixed β and p, vary r
# axa = Axis(fig[1,1], xgridvisible=false, ygridvisible=false, xlabel="ν", ylabel="c(ν)", limits=(nothing, (-0.05, 1.05)))
# β = 0.04; p = 0.6; rs = 10 .^ (range(-2.0, log10(2.0), 20))
# axa.title = "Fixed β = $(β), p = $(p)"
# Colorbar(fig[1,2], limits=extrema(rs), scale=log10, colormap=:viridis, label="r")
# for r in rs
#     lines!(axa, νs, [c(β, p, r, ν) for ν in νs], color=r, colorrange=extrema(rs), colorscale=log, colormap=:viridis)
# end

# # fixed β and r, vary p
# axb = Axis(fig[1,3], xgridvisible=false, ygridvisible=false, xlabel="ν", ylabel="c(ν)", limits=(nothing, (-0.05, 1.05)))
# β = 0.04; r = 0.25; ps = range(0.1, 1.0, 20)
# axb.title = "Fixed β = $(β), r = $(r)"
# Colorbar(fig[1,4], limits=extrema(ps), colormap=:viridis, label="p")
# for p in ps
#     lines!(axb, νs, [c(β, p, r, ν) for ν in νs], color=p, colorrange=extrema(ps), colormap=:viridis)
# end

# display(fig)

capped_hill(p, τ, h, ν) = p * (ν .^ h) ./ (τ^h .+ ν .^ h)

fig = Figure(size=(6inch, 2.5inch))
νs = 0:25

# fixed p and τ, vary h
axa = Axis(fig[1,1], xgridvisible=false, ygridvisible=false, xlabel="ν", ylabel="c(ν)", limits=(nothing, (-0.05, 1.05)))
p = 0.7; τ = 10; hs = 10 .^ (range(0.0, 2.0, 11))
axa.title = "Fixed p = $(p), τ = $(τ)"
Colorbar(fig[1,2], limits=extrema(hs), colormap=:viridis, scale=log10, label="h")
for h in hs
    lines!(axa, νs, capped_hill(p, τ, h, νs), color=h, colorrange=extrema(hs), colorscale=log10, colormap=:viridis)
end

# fixed p and h, vary τ
axb = Axis(fig[1,3], xgridvisible=false, ygridvisible=false, xlabel="ν", ylabel="c(ν)", limits=(nothing, (-0.05, 1.05)))
p = 0.7; h = 10; τs = range(5, 10, 6)
axb.title = "Fixed p = $(p), h = $(h)"
Colorbar(fig[1,4], limits=extrema(τs), colormap=:viridis, label="h")
for τ in τs
    lines!(axb, νs, capped_hill(p, τ, h, νs), color=τ, colorrange=extrema(τs), colormap=:viridis)
end

display(fig)