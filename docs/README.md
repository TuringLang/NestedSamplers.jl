For building the README we use [Weave.jl](https://github.com/JunoLab/Weave.jl) to process the [`README.jmd`](README.jmd) file. This provides a way for accurately portraying docstring changes and testing if examples run completely independently of the main Turing docs. 

Currently, I have a script set up with my git hooks for running Weave before pushing. Using it for every commit takes too long and I don't yet have a way for updating the README via CI (without awkward double-commits).

This script can be found at [`make`](make). You can choose to run this manually or you can use it as a git hook by moving it to the `.git/hooks` folder. To set it up to run before pushing simply run

    cp docs/make .git/hooks/pre-receive
