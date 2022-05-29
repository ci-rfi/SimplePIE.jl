using SimplePIE
using Documenter

DocMeta.setdocmeta!(SimplePIE, :DocTestSetup, :(using SimplePIE); recursive=true)

makedocs(;
    modules=[SimplePIE],
    authors="Chen Huang",
    repo="https://github.com/chenspc/SimplePIE.jl/blob/{commit}{path}#{line}",
    sitename="SimplePIE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chenspc.github.io/SimplePIE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chenspc/SimplePIE.jl",
    devbranch="main",
)
