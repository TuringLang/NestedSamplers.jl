using Documenter
using NestedSamplers

DocMeta.setdocmeta!(
    NestedSamplers,
    :DocTestSetup,
    :(using NestedSamplers);
    recursive=true
)

makedocs(
    sitename = "NestedSamplers.jl",
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "Gaussian Shells" => "examples/shells.md",
            "Correlated Gaussian" => "examples/correlated.md"
        ],
        "API/Reference" => "api.md"
    ],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [NestedSamplers],
    # https://github.com/JuliaLang/julia/pull/37085#issuecomment-683356098
    doctestfilters = [
        r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
        r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
        r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
    ]
)

deploydocs(repo = "github.com/TuringLang/NestedSamplers.jl.git", push_preview=true, devbranch="main")
