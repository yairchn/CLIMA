using Test, Pkg

@testset "Utilities" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["TicToc", "VariableTemplates", "ParametersType"]
        if all_tests || "$submodule" in ARGS || "Utilities" in ARGS
            include_test(submodule)
        end
    end

end
