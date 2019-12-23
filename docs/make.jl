using NestedSamplers
using Documenter

makedocs(;
    modules=[NestedSamplers],
    authors="Miles Lucas <mdlucas@hawaii.edu>",
    repo="https://github.com/mileslucas/NestedSamplers.jl/blob/{commit}{path}#L{line}",
    sitename="NestedSamplers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mileslucas.com/NestedSamplers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mileslucas/NestedSamplers.jl",
)
