module AttractorNetwork
export OnePopulation
export TwoPopulations
export ThreePopulations
export FourPopulations
export AdaptTwoPop
export AdaptFourPop
export FullSim
export FullSim_DitributedGains
export AdaptFullSim
export AdaptFullSimDistrGains

using PolynomialRoots
using HCubature
using SpecialFunctions
using Roots
using NLsolve
using Distributed, DataFrames, CSV
using Random
using StatsBase
using DelimitedFiles
using IterTools
using SparseArrays
using MAT

# Joint probabilities for correlated patterns
p1(α, β) = α * β + (1 - α) * (1 - β)
p1(p) = p1(p.α, p.β)
p11(α, β) = α * β^2 + (1 - α) * (1 - β)^2
p11(p) = p11(p.α, p.β)
p10(β::Number) = β * (1 - β)
p10(p) = p10(p.β)
p00(α, β) = α * (1 - β)^2 + (1 - α) * β^2
p00(p) = p00(p.α, p.β)

struct NetworkParameters
    γ::Float64
    C_hat::Float64
    A::Float64
    α::Float64
    β::Float64
end
function NetworkParameters(; γ = 1/1000, A = 1, C_hat)
    α, β = get_α_β(γ, C_hat)
    NetworkParameters(γ, C_hat, A, α, β)
end
function get_α_β(γ, C_hat)
    beta_roots = roots([ - γ * (1 - γ) * (1 - C_hat), (1 + 2 * γ * (1 - γ) * (1 - C_hat)), -3, 2])
    α_candidate = 0.
    β_candidate = 0.
    for β in beta_roots
        if imag(β) == 0 && real(β) >= 0 && real(β) <= 1.000000001
            β = real(β)
            α = (γ + β - 1)/(2 * β - 1)
            C_hat_recomputed=(p11(α, β) - γ^2)/(γ*(1-γ))
            γ_recomputed = p1(α, β)
            if α >= 0 && α <= 1.000000001 &&
                abs(C_hat_recomputed - C_hat) < 0.001 &&
                abs(γ_recomputed - γ) < 0.001
                #println("C_hat_recomputed = ", C_hat_recomputed, ", γ_recomputed = ", γ_recomputed)
                #if α_candidate != 0  # i.e. if \alpha_candidate is already assigned   
                #    @warn("Using second values of α and β. First values are $α_candidate and $β_candidate.")
                #end
                α_candidate = α
                β_candidate = β
            end
            #break # uncomment this if you want to use the first values of α and β 
        end
    end
    #println("α and β = are $α_candidate and $β_candidate.")
    return (α = α_candidate, β = β_candidate)
end

struct AdaptNetworkParameters
    γ::Float64
    C_hat::Float64
    A::Float64
    α::Float64
    β::Float64
    J0::Float64
end
function AdaptNetworkParameters(; γ = 1/1000, A = 1, C_hat, J0)
    α, β = get_α_β(γ, C_hat)
    AdaptNetworkParameters(γ, C_hat, A, α, β, J0)
end

# All we need for the gain functions
Base.@kwdef struct Heaviside
    h0::Float64 = 10.
    rm::Float64 = 50.
end
(h::Heaviside)(x) = h.rm .* 0.5 .* (sign.(x .- h.h0) .+ 1) 
struct Gain
    b::Float64 
    h0::Float64
    rm::Float64
end
function Gain(; b = 0.8, h0 = 10., rm = 50.)
    b >= 1000 && return Heaviside(h0, rm)
    Gain(b, h0, rm)
end
(g::Gain)(x) = g.rm ./ (1 .+ exp.(- g.b .* (x .- g.h0)))
gain(g, x) = g(x)
gain_squared(g, x) = g(x)^2
dgain_dx(g::Gain, x) = g.b * g.rm / (exp(g.b * (x - g.h0)) + exp(-g.b * (x - g.h0)) + 2)
function dgain_dx(g::Heaviside, x)
    x == g.h0 && return 1 #error("we are approximating a Dirac delta unproperly")  # It's only used in Fig 1.A
    0.
end

# Def the quenched noise term from the background patterns
struct Noise
    load::Float64
    r::Float64
    σ::Float64
end
Noise(; load, r, A) = Noise(load, r, A * sqrt(max(0, load * r)))

struct NoNoise end

struct NetworkParameters_FullSIM
    N::Int64
    P::Int64
    n::Int64  ## number of correlated signals
    γ::Float64
    C_hat::Float64
    A::Float64
    α::Float64
    β::Float64
    tau::Float64
end
function NetworkParameters_FullSIM(; N, P, n, γ = 1/1000, A = 1, C_hat, tau = 1)   
    α, β = get_α_β(γ, C_hat)
    NetworkParameters_FullSIM(N, P, n, γ, C_hat, A, α, β, tau)
end

function Reproduce_experiment(;γ = 0.002, c_hat = 0.04, N = 1000, n_repetitions = 8)
    C_hat = (c_hat - γ)/(1. - γ)
    N_groups_per_size = readdlm("Files/matrix_of_group_sizes_times_sections.dat")  # ones(98, 45) .+ ones(98, 45)  #
    responses = readdlm("Files/neuron_that_respond_per_session.dat")
    N_active = round(Int, γ * N)
    max_group_size = 45   # we consider a maximal sub-groups size of 45 (which is large enough for the chosen dataset)
    how_many_neurons_respond_to_how_many_concepts_parent = zeros(max_group_size, n_repetitions)
    how_many_neurons_respond_to_how_many_concepts_random = zeros(max_group_size, n_repetitions)
    how_many_neurons_respond_to_how_many_concepts_ind = zeros(max_group_size, n_repetitions)
    how_many_neurons_respond_to_how_many_concepts_strict = zeros(max_group_size, n_repetitions)
    
    for trial in 1:n_repetitions  # n_repetition is the number of times we repeat the experiment (e.g. in the paper, this parameter is set to 40)
        println( "###################### trial ", trial) 
        for s in 1:98             # s is the session number
            groups_in_this_session = round(Int,sum(N_groups_per_size[s, :]))
            indip_p_mat = zeros(N,  groups_in_this_session)

            #random.seed(s)   # does this work if uncommeted????
            n_neurons_to_sample = round(Int, responses[s])
            patterns_in_this_session = 0
            for n in 0:44
                patterns_in_this_session += round(Int,N_groups_per_size[s, n+1]*n)
            end 
            
            #println("patterns_in_this_session = ", patterns_in_this_session)
            pattern_matrix_parent = zeros(N, patterns_in_this_session)
            pattern_matrix_random = zeros(N, patterns_in_this_session)
            pattern_matrix_ind = zeros(N, patterns_in_this_session)
            pattern_matrix_strict = zeros(N, patterns_in_this_session)
            pattern_number = 1
            group_number = 0
            idx = 1
            for n in 0:44           # n is the number of the correlated patterns in the sub-group
                #println("n = ", n, " P = ", patterns_in_this_session, "   pattern_number = ", pattern_number)
                if N_groups_per_size[s, n+1] != 0
                    #@show s, n, N_groups_per_size[s, n+1]
                    for group in 1:N_groups_per_size[s, n+1] 
                        p = NetworkParameters_FullSIM( N = N, P = n, n = n, γ = γ, C_hat = C_hat)
                        # uncomment this for the sanity check
                        #open("Files/corr_pattern_matrix_N$(N)_session$(s)_group_size$(n).dat","w") do pat_mat_file  
                            pattern_matrix_random[:, pattern_number:(pattern_number+n-1)] = generate_random_binary_patterns_single_group(p)  
                            # here we create the histogram of correlations / uncomment if you want the create the histogram
                            #for i in 1:patterns_in_this_session
                            #    for j in i+1 : patterns_in_this_session
                            #        push!(shared_neus, pattern_matrix_random[:,i]' * pattern_matrix_random[:,j])
                            #    end
                            #end
                            p_mat =   generate_binary_patterns_single_group(p) #generate_parent_binary_patterns_fix_sparseness_single_group(p) #
                            pattern_matrix_parent[:, pattern_number:(pattern_number+n-1)] = p_mat
                            indip_p_mat[:, idx] = pattern_matrix_parent[:, pattern_number]
                            idx += 1
                            #writedlm(pat_mat_file, p_mat) # uncomment this for the sanity check
                            pattern_matrix_ind[:, pattern_number:(pattern_number+n-1)] = generate_indicators_binary_patterns_single_group(p)
                            pattern_matrix_strict[:, pattern_number:(pattern_number+n-1)] = generate_strict_parent_binary_patterns_single_group(p)
                            pattern_number += n
                            group_number += 1
                        #end
                    end
                end
            end 
            # uncomment this for the sanity check
            #open("Files/indip_pattern_matrix_N$(N)_session$(s).dat","w") do pat_mat_file_indip
            #    writedlm(pat_mat_file_indip, indip_p_mat)
            #end

            neu_that_responded_parent = 0
            while neu_that_responded_parent <=  n_neurons_to_sample
                neu = rand(1:N)
                n_responses_parent = round(Int, sum(pattern_matrix_parent[neu, :]))
                #println("n_responses = ",n_responses)
                if n_responses_parent != 0
                    neu_that_responded_parent += 1
                    how_many_neurons_respond_to_how_many_concepts_parent[n_responses_parent, trial] += 1
                end
            end
            neu_that_responded_random = 0
            while neu_that_responded_random <=  n_neurons_to_sample
                neu = rand(1:N)
                n_responses_random = round(Int, sum(pattern_matrix_random[neu, :]))
                #println("n_responses = ",n_responses)
                if n_responses_random != 0
                    neu_that_responded_random += 1
                    how_many_neurons_respond_to_how_many_concepts_random[n_responses_random, trial] += 1
                end
            end
            neu_that_responded_ind = 0
            while neu_that_responded_ind <=  n_neurons_to_sample
                neu = rand(1:N)
                n_responses_ind = round(Int, sum(pattern_matrix_ind[neu, :]))
                #println("n_responses = ",n_responses)
                if n_responses_ind != 0
                    neu_that_responded_ind += 1
                    #println(n_responses_ind)
                    if n_responses_ind <= max_group_size
                        how_many_neurons_respond_to_how_many_concepts_ind[n_responses_ind, trial] += 1
                    end
                end
            end
            neu_that_responded_strict = 0
            while neu_that_responded_strict <=  n_neurons_to_sample
                neu = rand(1:N)
                n_responses_strict = round(Int, sum(pattern_matrix_strict[neu, :]))
                #println("n_responses = ",n_responses)
                if n_responses_strict != 0
                    neu_that_responded_strict += 1
                    #println(n_responses_strict)
                    if n_responses_strict <= max_group_size
                        how_many_neurons_respond_to_how_many_concepts_strict[n_responses_strict, trial] += 1
                    end
                end
            end
        end
    end
    open("Files/strict_how_many_neurons_respond_to_how_many_concepts_N$(N)_trials$(n_repetitions).dat","w") do file_strict
        writedlm(file_strict, how_many_neurons_respond_to_how_many_concepts_strict)
    end
    open("Files/random_how_many_neurons_respond_to_how_many_concepts_N$(N)_trials$(n_repetitions).dat","w") do file_rnd
        writedlm(file_rnd, how_many_neurons_respond_to_how_many_concepts_random)
    end
    open("Files/parent_how_many_neurons_respond_to_how_many_concepts_N$(N)_trials$(n_repetitions).dat","w") do file_parent
        writedlm(file_parent, how_many_neurons_respond_to_how_many_concepts_parent)
    end
    open("Files/ind_how_many_neurons_respond_to_how_many_concepts_N$(N)_trials$(n_repetitions).dat","w") do file_ind    
        writedlm(file_ind, how_many_neurons_respond_to_how_many_concepts_ind)
    end
    #open("Files/corr_hist_N$(N).dat","w") do histo_file  
    #    writedlm(histo_file, shared_neus)
    #end
end

function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
end

function generate_random_binary_patterns_single_group(p)  
    if (p.P - p.n) < 0
        @error("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    #Random.seed!(420)
    items = [0, 1]
    pattern_matrix = zeros(p.N, p.n)
    c = p.C_hat * (1. - p.γ) + p.γ
    #@show c
    N_shared = round(Int, p.γ*c*p.N)
    N_active = round(Int, p.γ*p.N)
    #@show N_shared
    #@show N_active
    untouched_neus = Int64[]
    pattern_matrix[1:N_active,1] .= 1
    perm = copy(pattern_matrix[:,1])
    pattern_matrix[:,1] = perm[randperm(p.N)]
    for j in 1 : p.N
        #pattern_matrix[j,1] = sample(items, Weights([1 - p.γ, p.γ]))
        if pattern_matrix[j, 1] == 0
            push!(untouched_neus, j)
        end   
    end
    #@show  length(untouched_neus)

    #neus_not_to_double_pick = Int64[]
    for i in 2 : p.n
        #@show i
        for k in 1:i-1
            #identify neurons in common between pattern i and i-k
            number_of_neu_in_common = pattern_matrix[:,i-k]' * pattern_matrix[:,i]
            #@show  i, i-k
            #@show number_of_neu_in_common
            vect = Int64[]
            for j in 1:p.N
                if pattern_matrix[j,i-k] == 1  && pattern_matrix[j,i] == 0  
                    push!(vect, j)
                end
            end
            #this didn't work, because it excludes hub neurons
            #for j in neus_not_to_double_pick
            #    remove!(vect, j)
            #end

            #@show length(vect)
            permuted = vect[randperm(length(vect))]
            #@show N_shared - number_of_neu_in_common
            neus_in_common = permuted[1:round(Int,N_shared - number_of_neu_in_common)]
            #@show neus_in_common
            for j in neus_in_common
                pattern_matrix[round(Int,j),i] = 1 
            end
        end
        #@show sum(pattern_matrix[:,i])
        neurons_left = N_active - sum(pattern_matrix[:,i]) 
        if neurons_left < 0
            @show neurons_left
        end
        #@show neurons_left
        #@show length(untouched_neus)
        #perm = untouched_neus[randperm(length(untouched_neus))]
        neurons_to_assign =  untouched_neus[1:round(Int,neurons_left)] #perm[1:round(Int,neurons_left)]
        #@show neurons_to_assign
        #@show length(untouched_neus)-neurons_left
        for j in 1:length(neurons_to_assign)
            idx = round(Int,neurons_to_assign[j])
            pattern_matrix[idx, i] = 1
            #@show idx
            if pattern_matrix[idx,i-1] == 1
                @show idx
            end
            #@show idx in untouched_neus
            remove!(untouched_neus, idx)
            #@show idx in untouched_neus
        end 
    end 
    
    return pattern_matrix
end

function prob_parent_pattern(p, n1) # here we import the pattern matrix derived from theoretical probabilities contrsucted in python
    #alpha, beta = get_alpha_beta(gamma, C)
    n0 = p.n - n1
    prob = (p.α * p.β^n1 * (1-p.β)^n0 + (1. - p.α)*(1. - p.β)^n1 * p.β^n0)
    return prob
end
function generate_parent_binary_patterns_fix_sparseness_single_group(p) #be careful you generated the right matrix a priori with python #takes too much time
    if (p.P - p.n) < 0
        print("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    #random.seed(460)     #(3)
    # This is a function that generate binary pattrns with fixed number of active units
    # The number share active neurons between 2 patterns is "shared_neus" = C gamma (1 - gamma) + gamma ^2
    # The common active units are chosen to be the one with lowest index in both patterns
    N_active = round(Int, p.γ * p.N)
    shared_neus = round(Int, p.N * (p.C_hat * p.γ * (1. - p.γ) + p.γ^2))
    pattern_matrix = zeros(p.P, p.N)
    #@show p.n
    iterations = Iterators.product(Iterators.repeated(0:1, p.n)...)   #list(itertools.product([0, 1], repeat=n))
    lst = collect(iterations)
    #lst = np.flip(lst, axis = 0)
    #@show lst[1] ,  lst[2]
    n_combinations = length(lst)
    #@show n_combinations
    list_of_neus = collect(range(1,p.N, step=1)) 
    
    index_neu = 1
    for i in 1:n_combinations
        combo = lst[i]
        active_patt = Int64[]
        #println("combo = ", combo)
        for j in 1:p.n
            if combo[j] == 1
                push!(active_patt , j)
            end
        end
        #print 'active_patt', active_patt
        n1 = sum(combo)
        #println("n1 = ", n1)
        parent_prob = prob_parent_pattern(p, n1)
        #print 'prob', parent_prob, N * parent_prob
        n_neu_to_be_assigned = round(Int, p.N * parent_prob)
        #print 'n_neu_to_be_assigned',n_neu_to_be_assigned, 'index_neu', index_neu, 'index_neu+n_neu_to_be_assigned', min(index_neu+n_neu_to_be_assigned, N)
        for j in active_patt
            #for idx in range(index_neu, index_neu+n_neu_to_be_assigned):
            for  idx in list_of_neus[index_neu : min(index_neu+n_neu_to_be_assigned, p.N)]
                #print 'idx', idx, 'pattern', j
                pattern_matrix[j,idx] = 1.
            end
        end
        index_neu += n_neu_to_be_assigned
    end
    return pattern_matrix'  
end

function generate_binary_patterns_single_group(p)  
    if (p.P - p.n) < 0
        @error("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    #Random.seed!(3)
    items = [0, 1]
    weights = [1 - p.γ, p.γ]
    pattern_matrix = zeros(p.N, p.n)
    parent_pattern = zeros(p.N, 1)
    #println("p.α = ", p.α, ", p.β = ", p.β)
    for j in 1 : p.N
        parent_pattern[j] = sample(items, Weights([1 - p.α, p.α]))
    end
    for i in 1 : p.n
        #println("n = ", i)
        for j in 1 : p.N
            if parent_pattern[j]==1
                pattern_matrix[j,i]= sample(items, Weights([1 - p.β, p.β]))
            else
                pattern_matrix[j,i]= sample(items, Weights([p.β, 1 - p.β]))
            end 
        end 
    end 
    return pattern_matrix
end 

function generate_indicators_binary_patterns_single_group(p)  
    if (p.P - p.n) < 0
        @error("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    #Random.seed!(420)
    items = [0, 1]
    weights = [1 - p.γ, p.γ]
    pattern_matrix = zeros(p.N, p.n)
    parent_pattern = zeros(p.N, 1)
    #println("p.α = ", p.α, ", p.β = ", p.β)
    c = p.γ + (1. - p.γ) * p.C_hat
    ϵ = 0.5
    λ = (c*p.γ - p.γ^2)/((1-ϵ)^2 - 2*p.γ*(1-ϵ) +c*p.γ)
    
    p_flip = (p.γ - λ * (1 - ϵ))/(1 - λ)
    for j in 1 : p.N
        parent_pattern[j] = sample(items, Weights([1 - λ, λ]))
    end
    for i in 1 : p.n
        #println("n = ", i)
        for j in 1 : p.N
            if parent_pattern[j]==1
                pattern_matrix[j,i]= sample(items, Weights([0., 1 - ϵ]))
            else
                pattern_matrix[j,i]= sample(items, Weights([1 - p_flip, p_flip]))
            end 
        end 
    end 
    return pattern_matrix
end 

function generate_strict_parent_binary_patterns_single_group(p)  
    if (p.P - p.n) < 0
        @error("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    #Random.seed!(420)
    items = [0, 1]
    weights = [1 - p.γ, p.γ]
    pattern_matrix = zeros(p.N, p.n)
    parent_pattern = zeros(p.N, 1)
    c = p.C_hat * (1 - p.γ) + p.γ
    α = p.γ/c
    for j in 1 : p.N
        parent_pattern[j] = sample(items, Weights([1 - α, α]))
    end
    for i in 1 : p.n
        #println("n = ", i)
        for j in 1 : p.N
            if parent_pattern[j]==1
                pattern_matrix[j,i]= sample(items, Weights([1 - c, c]))
            end 
        end 
    end 
    return pattern_matrix
end 

# ------------------- end of single group ------------- beginning of full pattern matrices -------------------------
function generate_binary_patterns(p)   #randomly sampled or iterative model
    if (p.P - p.n) < 0
        @error("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    Random.seed!(420)
    items = [0, 1]
    weights = [1 - p.γ, p.γ]
    pattern_matrix = zeros(p.N, p.P)
    parent_pattern = zeros(p.N, 1)
    #println("p.α = ", p.α, ", p.β = ", p.β)
    for j in 1 : p.N
        parent_pattern[j] = sample(items, Weights([1 - p.α, p.α]))
    end
    for i in 1 : p.n
        println("n = ", i)
        for j in 1 : p.N
            if parent_pattern[j]==1
                pattern_matrix[j,i]= sample(items, Weights([1 - p.β, p.β]))
            else
                pattern_matrix[j,i]= sample(items, Weights([p.β, 1 - p.β]))
            end 
        end 
    end 
    for i in (p.n +1) : p.P
        for j in 1 : p.N
            pattern_matrix[j,i]=Int16(sample(items, Weights(weights)))
        end 
    end 
    println("generated binary patters, correlation = ", p.C_hat)
    return pattern_matrix
end 

function generate_2groups_binary_patterns_fix_sparseness(p, p1, p2) #shared neurns are always the first ones
    p_mat = zeros(Int, p.N, p.P)
    p_mat[:, 1 : p.n] = generate_binary_patterns_single_group(p1)  #generate_binary_patterns_fix_sparseness(p1)  #[:, 1:p.n]
    p_mat[:, p.n+1 : p.P] = generate_binary_patterns_single_group(p2) #generate_binary_patterns_fix_sparseness(p2)  #[:, p.n+1 : p.P]
    return p_mat
end

function generate_random_binary_patterns_2contex_3people(p) # used for the experimental prediction 
    if (p.P - p.n) < 0
        @error("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    Random.seed!(570)    #550
    items = [0, 1]
    pattern_matrix = zeros(p.N, 5) # we first store the 3 people (patterns 1 to 3) and then store the 2 contexes (patterns 4 and 5)
    c = p.C_hat * (1. - p.γ) + p.γ
    N_shared = round(Int, p.γ*c*p.N)
    N_active = round(Int, p.γ*p.N)
    
    untouched_neus = Int64[]
    pattern_matrix[1:N_active,1] .= 1
    perm = copy(pattern_matrix[:,1])
    pattern_matrix[:,1] = perm[randperm(p.N)]
    # pattern 1 is person 1
    for j in 1 : p.N
        if pattern_matrix[j, 1] == 0
            push!(untouched_neus, j)
        end   
    end
    # build person 2
    #identify neurons in common between pattern 2 and 1
    number_of_neu_in_common = pattern_matrix[:,1]' * pattern_matrix[:,2]
    vect = Int64[]
    for j in 1:p.N
        if pattern_matrix[j,1] == 1  && pattern_matrix[j,2] == 0  
            push!(vect, j)
        end
    end
    permuted = vect[randperm(length(vect))]
    #@show N_shared - number_of_neu_in_common
    neus_in_common = permuted[1:round(Int,N_shared - number_of_neu_in_common)]
    #@show neus_in_common
    for j in neus_in_common
        pattern_matrix[round(Int,j),2] = 1 
    end
    
    neurons_left = N_active - sum(pattern_matrix[:,2]) 
    if neurons_left < 0
        @show neurons_left
    end
    neurons_to_assign =  untouched_neus[1:round(Int,neurons_left)] 
    for j in 1:length(neurons_to_assign)
        idx = round(Int,neurons_to_assign[j])
        pattern_matrix[idx, 2] = 1
        remove!(untouched_neus, idx)
    end 
    # build person 3
    #identify neurons in common between pattern 3 and 1
    number_of_neu_in_common = pattern_matrix[:,1]' * pattern_matrix[:,3]
    vect = Int64[]
    for j in 1:p.N
        if pattern_matrix[j,1] == 1  && pattern_matrix[j,3] == 0  
            push!(vect, j)
        end
    end
    permuted = vect[randperm(length(vect))]
    #@show N_shared - number_of_neu_in_common
    neus_in_common = permuted[1:round(Int,N_shared - number_of_neu_in_common)]
    #@show neus_in_common
    for j in neus_in_common
        pattern_matrix[round(Int,j),3] = 1 
    end
    
    neurons_left = N_active - sum(pattern_matrix[:,3]) 
    if neurons_left < 0
        @show neurons_left
    end
    neurons_to_assign =  untouched_neus[1:round(Int,neurons_left)] 
    for j in 1:length(neurons_to_assign)
        idx = round(Int,neurons_to_assign[j])
        pattern_matrix[idx, 3] = 1
        remove!(untouched_neus, idx)
    end 
    
    # build concept C1 in common between P1 and P2
    for k in [1,2]
        #identify neurons in common between pattern 4 and k
        number_of_neu_in_common = pattern_matrix[:,k]' * pattern_matrix[:,4]
        vect = Int64[]
        for j in 1:p.N
            if pattern_matrix[j,k] == 1  && pattern_matrix[j,4] == 0  
                push!(vect, j)
            end
        end
        permuted = vect[randperm(length(vect))]
        #@show N_shared - number_of_neu_in_common
        neus_in_common = permuted[1:round(Int,N_shared - number_of_neu_in_common)]
        #@show neus_in_common
        for j in neus_in_common
            pattern_matrix[round(Int,j),4] = 1 
        end
    end
    neurons_left = N_active - sum(pattern_matrix[:,4]) 
    if neurons_left < 0
        @show neurons_left
    end
    neurons_to_assign =  untouched_neus[1:round(Int,neurons_left)] 
    for j in 1:length(neurons_to_assign)
        idx = round(Int,neurons_to_assign[j])
        pattern_matrix[idx, 4] = 1
        remove!(untouched_neus, idx)
    end 
    # build concept C2 in common between P1 and P3
    for k in [1,3]
        #identify neurons in common between pattern 5 and k
        number_of_neu_in_common = pattern_matrix[:,k]' * pattern_matrix[:,5]
        vect = Int64[]
        for j in 1:p.N
            if pattern_matrix[j,k] == 1  && pattern_matrix[j,5] == 0  
                push!(vect, j)
            end
        end
        permuted = vect[randperm(length(vect))]
        #@show N_shared - number_of_neu_in_common
        neus_in_common = permuted[1:round(Int,N_shared - number_of_neu_in_common)]
        #@show neus_in_common
        for j in neus_in_common
            pattern_matrix[round(Int,j),5] = 1 
        end
    end
    neurons_left = N_active - sum(pattern_matrix[:,5]) 
    if neurons_left < 0
        @show neurons_left
    end
    neurons_to_assign =  untouched_neus[1:round(Int,neurons_left)] 
    for j in 1:length(neurons_to_assign)
        idx = round(Int,neurons_to_assign[j])
        pattern_matrix[idx, 5] = 1
        remove!(untouched_neus, idx)
    end 
    
    return pattern_matrix
end

function generate_binary_patterns_fix_sparseness(p) # shared neurns are always the first ones
    if (p.P - p.n) < 0
        @error("Chosen value of correlated patterns p bigger than total number of patterns P")
    end
    Random.seed!(420)     #(3)
    # This is a function that generate binary pattrns with fixed number of active units
    # The number share active neurons between 2 patterns is "shared_neus" = C gamma (1 - gamma) + gamma ^2
    # The common active units are chosen to be the one with lowest index in both patterns
    N_active = ceil(Int, p.γ * p.N)
    shared_neus = round(Int, p.N * (p.C_hat * p.γ * (1 - p.γ) + p.γ ^2))
    xi_unsorted=zeros(Int, p.N, p.P)
    xi_unsorted[1:N_active,:] = ones(Int, N_active, p.P)
    for nu in 1 : p.n 
        last_part = xi_unsorted[(shared_neus + 1): p.N, nu]
        last_part =  last_part[randperm(p.N - shared_neus)]
        for i in 1 : p.N
            if i <= shared_neus
                xi_unsorted[i,nu] = 1
            elseif (i-shared_neus) <= (p.N - shared_neus)
                xi_unsorted[i,nu] = last_part[i-shared_neus]
            end
        end
    end  
    for nu in (p.n +1) : p.P
        xi_unsorted[:, nu]=xi_unsorted[randperm(p.N), nu]
    end   
    return xi_unsorted
end

# with fixed number of inputs per neuron
function diluted_weight_matrix(p, Pmatrix, dilution)
    # N = number of neurons
    # γ = mean activit of each pattern
    # Pmatrix = pattern matrix, size (N, P)
    println("dilution = ", dilution)
    Random.seed!(420) #420
    #items = [0, 1]
    #weights = [1. - dilution, dilution]
    K = Int(dilution*p.N) # Number of inputss

    c_matrix = zeros(p.N, p.N) #spzeros(p.N, p.N) #
    c_unsorted = zeros(Int, p.N)  #zeros(Int, p.N)
    c_unsorted[1:K] .= 1     #ones(Int, K)
    for i in 1 : p.N
        c_matrix[ i, :] = c_unsorted[randperm(p.N)]
    end   
    normalization = p.A / (dilution * p.N * p.γ * (1 - p.γ))   #dilution *
    #println("normalization = " , normalization)
    #W = sparse(normalization .* c_matrix .* ((Pmatrix .- p.γ) * (Pmatrix' .- p.γ)) )
    W = normalization .* c_matrix .* ((Pmatrix .- p.γ) * (Pmatrix' .- p.γ)) 
    for i in 1 : p.N   # the effect of taking self interaction away is negligible and it doesn't decrease the noise
       W[i,i] = 0
    end 
    #println("weight sum = " , sum(W))
    return W
end 

function  weight_matrix(p, Pmatrix)
    # N = number of neurons
    # γ = mean activit of each pattern
    # Pmatrix = pattern matrix, size (N, P)
    normalization = p.A / (p.N * p.γ * (1 - p.γ))  
    W = normalization .* ((Pmatrix .- p.γ) * (Pmatrix' .- p.γ)) 
    for i in 1 : p.N
        W[i,i] = 0
    end 

    return W
end 
function  input_field(p, g, p_mat, current_overlap, ext_input)  
    return p.A * g.rm * (p_mat .- p.γ) * current_overlap .+ ext_input
end 
function  overlap(p, pattern_matrix, S)  
    normalization = 1 ./(p.N * p.γ * (1 .- p.γ))
    return normalization .* ((pattern_matrix .- p.γ)' * S)
end

module OnePopulation
import ..AttractorNetwork: p1, NetworkParameters, Heaviside, Gain, dgain_dx, gain_squared, Noise, NoNoise
using PolynomialRoots
using HCubature
using SpecialFunctions
using Roots
using NLsolve
using Distributed, DataFrames, CSV

struct Results
    M1 :: Vector{Float64}  
    R :: Vector{Float64}  
end
Results() = Results([], [])

# Define the most used 4 inputs, no noise case
h1(p, g, m1) = p.A * g.rm * (1 - p.γ) * m1  
h0(p, g, m1) = -p.γ * p.A * g.rm * m1 

# Here we implement the 4 entries of the Jacobian matrix
function J1(m1, p, g, n::NoNoise) 
    - 1 + p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p.γ * dgain_dx(g, h1(p, g, m1)) + 
        p.γ^2 * (1-p.γ) * dgain_dx(g, h0(p, g, m1)))
end
function J1(m1, p, g, n::Noise) 
    integrand = x -> p.A * exp(-x[1]^2 / 2) / (sqrt(2 * pi) * p.γ * (1 - p.γ)) * ( 
        (1 - p.γ)^2 * p.γ * dgain_dx(g, h1(p, g, m1)+ n.σ*x[1]) + 
        p.γ^2 * (1-p.γ) * dgain_dx(g, h0(p, g, m1)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = -1 + hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_eigenvalues(p, g, n, m1)
    J1(m1, p, g, n)
end 

# System's variables: m1, p, q
function nullcline1(p, g, m1)
    1 / g.rm * (g(h1(p, g, m1))  -  g(h0(p, g, m1)))
end
function nullcline1(p, g::Gain, n::Noise, m1)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi) * g.rm * p.γ * (1 - p.γ)) * (
    (1 - p.γ) * p.γ * g(h1(p, g, m1)+ n.σ*x[1])  -
    p.γ * (1-p.γ) * g(h0(p, g, m1)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function nullcline1(p, g::Heaviside, n::Noise, m1)
    0.5 * erfc((g.h0 - h1(p, g, m1))/(sqrt(2)* n.σ)) - 0.5 * erfc((g.h0  -  h0(p, g, m1))/(sqrt(2)*n.σ))
end

function compute_q(p, g::Gain, n::Noise, m1) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p.γ * dgain_dx(g, h1(p, g, m1)+ n.σ*x[1]) + 
        (1-p.γ) * dgain_dx(g, h0(p, g, m1)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_q(p, g::Heaviside, n::Noise, m1)
    result = g.rm/sqrt(2*pi) * (p.γ * exp(- ((g.h0 - h1(p, g, m1))/(sqrt(2)*n.σ))^2)  +
        (1-p.γ) * exp(-((g.h0 + h0(p, g, m1))/(sqrt(2)*n.σ))^2))
    return result
end 

function compute_p(p, g::Gain, n::Noise, m1) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p.γ * gain_squared(g, h1(p, g, m1)+ n.σ*x[1]) + 
        (1-p.γ) * gain_squared(g, h0(p, g, m1)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_p(p, g::Heaviside, n::Noise, m1)
    result = g.rm^2 * (p.γ * 0.5 * erfc((g.h0 - h1(p, g, m1))/(sqrt(2)*n.σ))  +
        (1-p.γ) * 0.5 * erfc((g.h0 + h0(p, g, m1))/(sqrt(2)*n.σ)))
    return result
end

function get_fixed_points(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
    m_val = range( bound_low, stop= bound_up, length = resolution)
    r_val = range( 0, stop= bound_r, length = resolution)
    @show load
    results = Results()
    for l in 1 : resolution
        m1_temp = Float64[]
        if load != 0.
            for i in 1 : resolution
                n_temp = Noise(load = load, r = r_val[l], A = p.A)
                q_func = compute_q(p, g, n_temp, m_val[i])
                p_func = compute_p(p, g, n_temp, m_val[i]) 
                D_const = (1 - p.A * q_func)
                r_fin = p_func /(D_const^2)
                
                if abs(r_fin - r_val[l]) < bound_r /(resolution_factor * resolution)
                    if r_val[l] != 0.0            
                        m1 = nullcline1(p, g, n_temp, m_val[i])
                        push_to_fixed_point!(results, m1, m_val[i], m1_temp,  bound_up, bound_low, resolution_factor, resolution, r_val[l])
                        
                    else 
                        m1 = nullcline1(p, g, m_val[i])
                        push_to_fixed_point!(results, m1, m_val[i], m1_temp, bound_up, bound_low, resolution_factor, resolution, 0.)
                    end
                    continue
                end
            end
        else 
            m1 = nullcline1(p, g, m_val[i])
            push_to_fixed_point!(results, m1, m_val[i], m1_temp, bound_up, bound_low, resolution_factor, resolution, 0.)
        end
    end
    return  results.M1, results.R  
end 

function push_to_fixed_point!(results, m1, m_val1, m1_temp, bound_up, bound_low, resolution_factor, resolution, r)
    push!(m1_temp, m1 - m_val1)  
    if length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0  #abs(m1-m_val1)<(bound_up-bound_low)/(resolution_factor*resolution) #
        push!(results.M1, m_val1)
        push!(results.R, r)
    end
end

function get_fixed_points_and_nullclines(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
    m_val = range(bound_low, stop= bound_up, length = resolution)
    r_val=range(0, stop= bound_r, length = resolution)
    M11=Float64[] # m1 projection of m1 nullclin
    MR=Float64[]
    RM=Float64[]
    X_p=Float64[]
    R_p=Float64[]
    Stab=Float64[]
    R_temp=Float64[]
    for k in 1:resolution
        m1_temp = Float64[]
        for i in 1:resolution
            n_temp = Noise(load = load, r = r_val[k], A = p.A)
            q_func = compute_q(p, g, n_temp, m_val[i])
            p_func = compute_p(p, g, n_temp, m_val[i]) 
            D_const = (1 - p.A * q_func)
            r_fin = p_func /(D_const^2)
            
            if abs(r_fin-r_val[k])<(bound_r)/(resolution_factor*resolution)
                push!(R_temp, r_val[k])  
                push!(RM, m_val[i])  
            end
            m1 = nullcline1(p, g, n_temp, m_val[i])
            push!(m1_temp, m1 - m_val[i])
            if length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0  #abs(m1-m_val[i])<(bound_up-bound_low)/(resolution_factor*resolution)
                push!(M11, m1)
                push!(MR, r_val[k])
            end
        end
    end
    
    for l in 1:length(M11)
        for rr in 1:length(R_temp)
            if (abs(M11[l]-RM[rr])<=(bound_up-bound_low)/(resolution_factor*resolution)) && (abs(MR[l]-R_temp[rr])<=(bound_r)/(resolution_factor*resolution)) 
                n = Noise(load = load, r = RM[rr], A = p.A)
                stability = J1(M11[l], p, g, n) 
                push!(X_p, M11[l])
                push!(R_p, R_temp[rr])
                push!(Stab, stability)
            end
        end
    end
    
    println("length(M11)  ",length(M11))
    return  M11, MR,R_temp, RM, X_p,R_p, Stab
end

#check if this function works!!!!
function compute_critical_capacity(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, max_load = 10, resolution, resolution_factor = 1, size, c_hat = 0)
    g = Gain( rm = rm, b = b, h0 = h0 )
    load_range = range( 0.01, stop = max_load , length = size )
    open("Output_1pop/bifurcation_1pop_fixed_points_vs_load_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_chat$(c_hat).dat","w") do capacity
    for i in 1:size
        start = time()
        p = NetworkParameters(γ = γ , A = A, C_hat = c_hat)
        load = load_range[i]
        println( "load = ", load, "   h0 = ", h0)
        
        M1, R = get_fixed_points(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
        
        for j in 1: length(M1)
            n = NoNoise()
            if R[j] != 0.
                n = Noise(load = load, r = R[j], A = A)
            end
            eigs = real(compute_eigenvalues(p, g, n, M1[j]))
            println("eigenvalue = ",eigs[1], ", m = ", M1[j], ", r = ", R[j])
            if eigs[1] > 0 
                write(capacity, "$(load) ,  $(M1[j]) , $(R[j]) , $(1)  \n")
            else 
                write(capacity, "$(load) ,  $(M1[j]) , $(R[j]) , $(3)  \n")
            end
        end
        
        elapsed = time() - start
        println("Simulation of point ",  i ," of ", size, " took ", elapsed, " seconds")
    end
    end
end

# this function creates the phase-plane in the space m^1 - R
function PP_m1R(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, load = 10, resolution, resolution_factor = 1, c_hat = 0)
    g = Gain( rm = rm, b = b, h0 = h0 )
    p = NetworkParameters(γ = γ , A = A, C_hat = c_hat)
    M11, MR,R_temp, RM, X_p,R_p, Stab = get_fixed_points_and_nullclines(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
    open("Output_1pop/nullclineM_1pop_noise_gamma$(γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_load$(load).dat","w") do null
        for j in 1:length(M11)
            write(null, "$(M11[j])   $(MR[j]) \n")
        end
    end
    open("Output_1pop/nullclineR_1pop_noise_gamma$(γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_load$(load).dat","w") do null
        for j in 1:length(RM)
            write(null, "$(RM[j])   $(R_temp[j]) \n")
        end
    end

    open("Output_1pop/fp3D_1pop_noise_gamma$(γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_load$(load).dat","w") do fp 
        for j in 1: length(X_p)
            write(fp, "$(X_p[j])     $(R_p[j])   \n")
        end
    end   
    open("Output_1pop/Stab_1pop_noise_gamma$(γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_load$(load).dat","w") do stab 
        for j in 1: length(X_p)
            write(stab, "$(Stab[j])   \n")
        end
    end           
end

end  # end of module OnePopulation

module TwoPopulations
import ..AttractorNetwork: p1, p11, p10, p00, NetworkParameters, get_α_β, Heaviside, Gain, dgain_dx, gain_squared, Noise, NoNoise
using PolynomialRoots
using HCubature
using SpecialFunctions
using Roots
using NLsolve
using Distributed, DataFrames, CSV
using LinearAlgebra
using DelimitedFiles
# Define the 2 possible models
struct MeanField end
struct MFnoSelfInteraction end

# Define the most used 4 inputs, no noise case
h11(p, g, m1, m2, I1, I2) = p.A * g.rm * (1 - p.γ) * (m1 + m2) + I1 + I2
h00(p, g, m1, m2, I1, I2) = - p.γ * p.A * g.rm * (m1 + m2)
h10(p, g, m1, m2, I1, I2) = p.A * g.rm * (1 - p.γ) * m1 - p.γ * p.A * g.rm * m2 + I1
h01(p, g, m1, m2, I1, I2) = h10(p, g, m2, m1, I2, I1)

# Here we implement the 4 entries of the Jacobian matrix
function J11(m1, m2, I1, I2, p, g, n::NoNoise) 
    - 1 + p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p11(p) * dgain_dx(g, h11(p, g, m1, m2, I1, I2)) + 
        (1 - p.γ)^2 * p10(p) * dgain_dx(g, h10(p, g, m1, m2, I1, I2)) +
        p.γ^2 * p10(p) * dgain_dx(g, h01(p, g, m1, m2, I1, I2)) +
        p.γ^2 * p00(p) * dgain_dx(g, h00(p, g, m1, m2, I1, I2)))
end
function J12(m1, m2, I1, I2, p, g, n::NoNoise) 
    p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p11(p) * dgain_dx(g, h11(p, g, m1, m2, I1, I2)) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g, h10(p, g, m1, m2, I1, I2)) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g, h01(p, g, m1, m2, I1, I2)) +
        p.γ^2 * p00(p) * dgain_dx(g, h00(p, g, m1, m2, I1, I2)))
end
function J11(m1, m2, I1, I2, p, g, n::Noise) 
    integrand = x -> p.A * exp(-x[1]^2 / 2) / (sqrt(2 * pi) * p.γ * (1 - p.γ)) * ( 
        (1 - p.γ)^2 * p11(p) * dgain_dx(g, h11(p, g, m1, m2, I1, I2)+ n.σ*x[1]) + 
        (1 - p.γ)^2 * p10(p) * dgain_dx(g, h10(p, g, m1, m2, I1, I2)+ n.σ*x[1]) +
        p.γ^2 * p10(p) * dgain_dx(g, h01(p, g, m1, m2, I1, I2)+ n.σ*x[1]) +
        p.γ^2 * p00(p) * dgain_dx(g, h00(p, g, m1, m2, I1, I2)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = -1 + hcubature(integrand, -bound, bound)[1]
    return result
end
function J12(m1, m2, I1, I2, p, g, n::Noise) 
    integrand = x -> p.A * exp(-x[1]^2 / 2) / (sqrt(2 * pi) * p.γ * (1 - p.γ)) * ( 
        (1 - p.γ)^2 * p11(p) * dgain_dx(g, h11(p, g, m1, m2, I1, I2)+ n.σ*x[1]) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g, h10(p, g, m1, m2, I1, I2)+ n.σ*x[1]) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g, h01(p, g, m1, m2, I1, I2)+ n.σ*x[1]) +
        p.γ^2 * p00(p) * dgain_dx(g, h00(p, g, m1, m2, I1, I2)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
J22(m1, m2, I1, I2, p, g, n) = J11(m2, m1, I2, I1, p, g, n)
J21(m1, m2, I1, I2, p, g, n) = J12(m2, m1, I2, I1, p, g, n)

function eigenvalues(m1, m2, I1, I2, p, g, n)
    J = [J11(m1, m2, I1, I2, p, g, n)  J12(m1, m2, I1, I2, p, g, n)   ; 
        J21(m1, m2, I1, I2, p, g, n)  J22(m1, m2, I1, I2, p, g, n)]
    return eigvals(J)
end 

# System's variables: m1, m2, p, q
function nullcline1(p, g, n::NoNoise, h11, h10, h01, h00)
    1 / (g.rm * p.γ * (1 - p.γ)) * ((1 - p.γ) * p11(p.α, p.β) * g(h11) + 
        (1 - p.γ) * p10(p.β) * g(h10) -
        p.γ * p10(p.β) * g(h01) -
        p.γ * p00(p.α, p.β) * g(h00))
end
function nullcline1(p, g::Gain, n::Noise, h11, h10, h01, h00)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi) * g.rm * p.γ * (1 - p.γ)) * ( 
        (1 - p.γ) * p11(p.α, p.β) * g(h11 + n.σ*x[1]) + 
        (1 - p.γ) * p10(p.β) * g(h10 + n.σ*x[1]) -
        p.γ * p10(p.β) * g(h01 + n.σ*x[1]) -
        p.γ * p00(p.α, p.β) * g(h00 + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

denominator(p, n) = sqrt(2)* n.σ
recurrent_correction_Heaviside(p, g::Heaviside, n) = 0.5 * p.A * n.load * g.rm ## double check with the supplementary material!!! in particular check if there are the prob in front and if not, modify also the rest of the code
function nullcline1(p, g::Heaviside, n::Noise, h11, h10, h01, h00)
    result = 1. / (p.γ * (1 - p.γ)) * 0.5 * (
        (1 - p.γ) * p11(p.α, p.β) * erfc((g.h0 - h11 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) +
        (1 - p.γ) * p10(p.β) * erfc((g.h0 - h10 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) -
        p.γ * p10(p.β) * erfc((g.h0 - h01 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) -
        p.γ * p00(p.α, p.β) * erfc((g.h0 - h00 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) )
    return result
end
nullcline2(p, g, n, h11, h10, h01, h00) = nullcline1(p, g, n, h11, h01, h10, h00)

function solve_q(m1, m2, p, g, n, h11, h10, h01, h00) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * ( 
                    p11(p.α, p.β) * dgain_dx(g, h11 + n.σ*x[1]) +
                    p10(p.β) * dgain_dx(g, h10 + n.σ*x[1]) +
                    p10(p.β) * dgain_dx(g, h01 + n.σ*x[1]) +
                    p00(p.α, p.β) * dgain_dx(g, h00 + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function f!(F, h, m1, m2, I1, I2, p, g::Gain, n::Noise)
    F[1] = solve_q(m1, m2, p, g::Gain, n::Noise, h[2], h[3], h[4], h[5]) - h[1]
    F[2] = h11(p, g, m1, m2, I1, I2) + p.A^2 * n.load * h[1] * g(h[2]) / (1 - p.A * h[1]) - h[2]
    F[3] = h10(p, g, m1, m2, I1, I2) + p.A^2 * n.load * h[1] * g(h[3]) / (1 - p.A * h[1]) - h[3]
    F[4] = h01(p, g, m1, m2, I1, I2)  + p.A^2 * n.load * h[1] * g(h[4]) / (1 - p.A * h[1]) - h[4]
    F[5] = h00(p, g, m1, m2, I1, I2)  + p.A^2 * n.load * h[1] * g(h[5]) / (1 - p.A * h[1]) - h[5]
end
function compute_q( p, g::Gain, n::Noise, m1, m2, I1, I2, model::MFnoSelfInteraction)
    r= nlsolve((F, h) -> f!(F, h, m1, m2, I1, I2, p, g, n), [0.,0.,0.,0.,0. ])
    r.zero
end
function compute_q(p, g::Gain, n::Noise, m1, m2, I1, I2, model::MeanField) 
    h11_val = h11(p, g,  m1, m2, I1, I2)
    h10_val = h10(p, g, m1, m2, I1, I2)
    h01_val = h01(p, g, m1, m2, I1, I2)
    h00_val = h00(p, g, m1, m2, I1, I2)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * ( 
                    p11(p.α, p.β) * dgain_dx(g, h11_val + n.σ*x[1]) +
                    p10(p.β) * dgain_dx(g, h10_val + n.σ*x[1]) +
                    p10(p.β) * dgain_dx(g, h01_val + n.σ*x[1]) +
                    p00(p.α, p.β) * dgain_dx(g, h00_val + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result, h11_val, h10_val, h01_val, h00_val
end
function compute_q(p, g::Heaviside, n::Noise, m1, m2, I1, I2, model)
    h11_val = h11(p, g,  m1, m2, I1, I2)
    h10_val = h10(p, g, m1, m2, I1, I2)
    h01_val = h01(p, g, m1, m2, I1, I2)
    h00_val = h00(p, g, m1, m2, I1, I2)
    result = g.rm/sqrt(2*pi) * (p11(p) * exp(- ((g.h0 - h11_val)/(sqrt(2)*n.σ))^2)  +
    p10(p) * exp(- ((g.h0 - h10_val)/(sqrt(2)*n.σ))^2) +
    p10(p) * exp(- ((g.h0 - h01_val)/(sqrt(2)*n.σ))^2) +
    p00(p) * exp(- ((g.h0 - h00_val)/(sqrt(2)*n.σ))^2))
    return result, h11_val, h10_val, h01_val, h00_val
end 

function compute_p(p, g::Gain, n::Noise, h11, h10, h01, h00, model) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * ( 
                    p11(p.α, p.β) * gain_squared(g, h11 + n.σ*x[1]) +
                    p10(p.β) * gain_squared(g, h10 + n.σ*x[1]) +
                    p10(p.β) * gain_squared(g, h01 + n.σ*x[1]) +
                    p00(p.α, p.β) * gain_squared(g, h00 + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_p(p, g::Heaviside, n::Noise, h11, h10, h01, h00, model) 
    result = g.rm^2 * 0.5 * (p11(p.α, p.β) * erfc((g.h0 - h11 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) + 
        p10(p.β) * erfc((g.h0 - h10 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) +
        p10(p.β) * erfc((g.h0 - h01 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) + 
        p00(p.α, p.β) * erfc((g.h0 - h00 + recurrent_correction_Heaviside(p, g, n)) / denominator(p, n)) )
    return result
end

function push_to_nullcline!(m1_temp, m2_temp, m1, m2 , m_valj, m_vali, r_valk, results, bound_up, bound_low, resolution_factor, resolution)
    push!(m1_temp, m1 - m_valj)  
    if (length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0 ) #|| (abs(m1-m_valj)<(bound_up-bound_low)/(resolution_factor*resolution))
        push!(results.M11, m_valj)
        push!(results.M21, m_vali)
        push!(results.R, r_valk)
    end
    if (length(m2_temp) > 1 &&  m2_temp[end]*m2_temp[end-1] < 0 ) #|| (abs(m2-m_vali)<(bound_up-bound_low)/(resolution_factor*resolution))
        push!(results.M12, m_valj)
        push!(results.M22, m_vali)
    end
end 

struct Results
    M11 :: Vector{Float64} # m1 projection of m1 nullcline
    M21 :: Vector{Float64} # m2 projection of m1 nullcline
    M12 :: Vector{Float64} # m1 projection of m2 nullcline
    M22 :: Vector{Float64} # m2 projection of m2 nullcline
    R :: Vector{Float64}  #R coordinate of the first nullcline
end
Results() = Results([], [], [], [], [])

function get_nullclines(p, g, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, model, corr_noise)
    m_val = range( bound_low, stop= bound_up, length = resolution)
    r_val = range( 0, stop= bound_r, length = resolution)
    results = Results()
    open("Output_2pop/Quiver_PP_gamma$(p.γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(g.rm)_A$(p.A)_h0$(g.h0)_b$(g.b)_load$(load)_C_hat$(p.C_hat)_I1$(I1).dat","w") do quiver
    m2_temp = Float64[]
    m2 = 0
    for i in 1 : resolution
        m1_temp = Float64[]
        for j in 1 : resolution
            if load != 0.
                for k in 1 : resolution
                    n_temp = Noise(load=load, r=r_val[k], A = p.A)
                    q_func, h11_val, h10_val, h01_val, h00_val = compute_q(p, g, n_temp, m_val[j], m_val[i], I1, I2 , model)
                    p_func = compute_p(p, g, n_temp, h11_val, h10_val, h01_val, h00_val, model) 
                    D_const = (1 - p.A * q_func)
                    r_fin = p_func /(D_const^2)
                    r_fin1 = 0
                    if corr_noise == true
                        C_const = p.A * q_func * (p11(p.α, p.β) - p.γ^2) / (p.γ * (1 - p.γ))
                        r_fin = p_func * ( p.γ * (1 - p.γ) * ((D_const^2 + C_const^2) * p.γ * (1 - p.γ) + 2*D_const*C_const* (p11(p.α, p.β) - p.γ^2)) + (p11(p.α, p.β) - p.γ^2)*( 2*D_const*C_const * p.γ*(1 - p.γ) + (D_const^2 + C_const^2) * (p11(p.α, p.β) - p.γ^2))) / ((D_const^2 - C_const^2) *p.γ*(1 - p.γ))^2
                    end
                    if abs(r_fin - r_val[k]) < bound_r /(resolution_factor * resolution)
                        if r_val[k] != 0.0            
                            m1 = nullcline1(p, g, n_temp, h11_val, h10_val, h01_val, h00_val)
                            m2 = nullcline2(p, g, n_temp, h11_val, h10_val, h01_val, h00_val)

                            input_no_correction =  p00(p.α, p.β) * g(h00_val)
                            correction_term = (p.A^2 * p.γ * load *q_func ) / (4*( 1 - p.A * q_func ))
                            input_with_correction = p00(p.α, p.β) * g( (h00_val + correction_term)/(1 - correction_term))
                            push_to_nullcline!(m1_temp, m2_temp,  m1 , m2, m_val[j], m_val[i], r_fin, results, bound_up, bound_low, resolution_factor, resolution)
                        else 
                            n0 = NoNoise()
                            h11_var = h11(p, g, m_val[j], m_val[i], I1, I2)
                            h10_var = h10(p, g, m_val[j], m_val[i], I1, I2)
                            h01_var = h01(p, g, m_val[j], m_val[i], I1, I2)
                            h00_var = h00(p, g, m_val[j], m_val[i], I1, I2)
                            m1 = nullcline1( p, g, n0, h11_var, h10_var, h01_var, h00_var)
                            m2 = nullcline2( p, g, n0, h11_var, h10_var, h01_var, h00_var)
                            push_to_nullcline!(m1_temp, m2_temp,  m1 ,m2,   m_val[j], m_val[i], 0., results, bound_up, bound_low, resolution_factor, resolution)
                        end
                        continue
                    end
                end
            else 
                n0 = NoNoise()
                h11_var = h11(p, g, m_val[j], m_val[i], I1, I2)
                h10_var = h10(p, g, m_val[j], m_val[i], I1, I2)
                h01_var = h01(p, g, m_val[j], m_val[i], I1, I2)
                h00_var = h00(p, g, m_val[j], m_val[i], I1, I2)
                m1 = nullcline1( p, g, n0, h11_var, h10_var, h01_var, h00_var)
                m2 = nullcline2( p, g, n0, h11_var, h10_var, h01_var, h00_var)
                push_to_nullcline!(m1_temp,m2_temp,  m1 , m2, m_val[j], m_val[i], 0., results, bound_up, bound_low, resolution_factor, resolution)
            end
            if i%100 == 0 && j%100 == 0
                write(quiver, "$(m_val[j])   $(m_val[i])   $(m1 - m_val[j])  $(m2 - m_val[i])  \n")
            end
        end
        push!(m2_temp, m2 - m_val[i])
    end
    end
    
    return  results.M11, results.M21, results.M12, results.M22, results.R
end 

function get_fixed_points(M11, M12, M21, M22, R,  bound_up, bound_low, resolution, resolution_factor)
    # the coodinates of the fixed points
    X_p = Float64[]
    Y_p = Float64[]
    R_p = Float64[]
    for l in 1:length(M11)
        for m in 1:length(M12)
            if (abs(M12[m]-M11[l])<=(bound_up-bound_low)/(resolution_factor*resolution)) && (abs(M22[m]-M21[l])<=(bound_up-bound_low)/(resolution_factor*resolution))  
                push!(X_p, M11[l])
                push!(Y_p, M22[m])
                push!(R_p, R[m])
            end
        end
    end
    return X_p, Y_p, R_p
end

function get_fp_stability(X_p, Y_p, R_p, I1, I2, p, g, load)
    stab = Float64[]
    for j in 1: length(X_p)
        n = NoNoise()
        if R_p[j] != 0.
            n = Noise(load = load, r = R_p[j], A = p.A)
        end
        eigs=real( eigenvalues(X_p[j], Y_p[j], I1, I2, p, g, n))
        if eigs[1] > 0 && eigs[2] > 0
            push!(stab, 0)
        elseif (eigs[1] >= 0 && eigs[2] <= 0) || (eigs[1] <= 0 && eigs[2] >= 0)
            push!(stab, 1)
        elseif eigs[1] < 0 && eigs[2] < 0
            push!(stab, 2)
        end
    end
    return stab
end

### NOTE FOR A MORE POLISHED VERSION OF THE CODE: maybe take this function out and make it generic, so that we can use it for all 2 to 4 pop solutions.!!!!
function extract_critical_C_hat(rm, A, h0, γ, b, bound_low, bound_up, bound_r, resolution, resolution_factor, size, load, model, corr_noise)
    (data_cells, header_cells) = readdlm("Output_2pop/Bifurcation_plot_vs_C_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_model$(model)_corr_noise$(corr_noise).dat", header = true) 
    C_hat = data_cells[:, 1]
    x_p = data_cells[:, 2]
    r_p = data_cells[:, 3]
    stab = data_cells[:, 4]
    C_hat_crit = 0.
    for i in 1:length(x_p)
        if x_p[i] > 0.9 && stab[i] == 1.0
            C_hat_crit = C_hat[i]
        end
    end
    return C_hat_crit
end

function generate_bifurcation_diagram(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, I1 = 0., I2 = 0., bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution, resolution_factor = 1, size, load, model, corr_noise = false)
    g = Gain( rm = rm, b = b, h0 = h0 )
    c_hat_range = range( 0., stop = h0/rm + h0/(10*rm)  , length = size )  
    result = @distributed vcat for i in 1:size
        tmp = DataFrame(C_hat = [], x_p = [], r_p = [], stab = [])
        start = time()
        p = NetworkParameters(γ = γ , A = A, C_hat = c_hat_range[i])
        sum_probs = p11(p) + 2 * p10(p) + p00(p)

        M11, M21, M12, M22, R1 = get_nullclines(p, g, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, model, corr_noise)
        X_p, Y_p, R_p = get_fixed_points(M11, M21, M21, M11, R1, bound_up, bound_low, resolution, resolution_factor)
        stab = get_fp_stability(X_p, Y_p, R_p, I1, I2, p, g, load)
        for j in 1: length(X_p)
            push!(tmp, [p.C_hat, X_p[j], R_p[j],  stab[j]])
        end
        elapsed = time() - start
        println("Computing point ",  i ," of ", size, " took ", elapsed, " seconds")
        tmp
    end
    CSV.write("Output_2pop/Bifurcation_plot_vs_C_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_model$(model)_corr_noise$(corr_noise).dat", result, delim = ' ')

    crit_corr = extract_critical_C_hat(rm, A, h0, γ, b, bound_low, bound_up, bound_r, resolution, resolution_factor, size, load, model, corr_noise)
    println("Critical correlation = ", crit_corr)
end

function generate_critical_corr_vs_b_h0(; rm = 50, A = 1, γ = 0.001 , bound_low = -0.05, bound_up = 1.05, bound_r = 0.03, resolution, resolution_factor = 1, size, load, model, corr_noise = false)
    min_h0 = 0.1
    max_h0 = 1.2
    resolution_h0 = 100
    min_b = 0.
    max_b = 200.
    resolution_b = 200
    i = 0
    open("Output_2pop/critical_corr_vs_b_h0_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_load$(load).dat","w") do out_file
        for h0 in range( min_h0, stop = max_h0 , length = resolution_h0 )
            for b in range( min_b, stop = max_b , length = resolution_b )
                generate_bifurcation_diagram(rm = rm, A = A, h0 = h0, γ = γ, b = b, bound_low = bound_low, bound_up = bound_up, bound_r = bound_r, resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, model = model, corr_noise = corr_noise)
                C_hat_critical = extract_critical_C_hat(rm, A, h0, γ, b, bound_low, bound_up, bound_r, resolution, resolution_factor, size, load, model, corr_noise)
                write(out_file, "$(C_hat_critical)   ")
            end
            write(out_file, "\n")
            println("point ", i, " of ", resolution_h0 )
            i += 1
        end 
    end
end

# the next functions serve to generate the dynamiss in the PP    ### NOTE FOR A MORE POLISHED VERSION OF THE CODE: here there is a lot to optimize!!!
function update_state(p, g, m1_old, m2_old, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, running_time, h, t_onset, t_offset, tau_m, model, corr_noise)
    r_val = range( 0, stop= bound_r, length = resolution) 
    results = Results()
    m1 = 0.
    m2 = 0.
    r = 0.
    if load != 0. 
        for k in 1 : resolution
            n_temp = Noise(load=load, r=r_val[k], A = p.A)
            q_func, h11_val, h10_val, h01_val, h00_val = compute_q(p, g, n_temp, m1_old, m2_old, I1, I2 , model)
            p_func = compute_p(p, g, n_temp, h11_val, h10_val, h01_val, h00_val, model) 
            D_const = (1 - p.A * q_func)
            r_fin = p_func /(D_const^2)
            if corr_noise == true
                C_const = p.A * q_func * (p11(p.α, p.β) - p.γ^2) / (p.γ * (1 - p.γ))
                r_fin = p_func * ( p.γ * (1 - p.γ) * ((D_const^2 + C_const^2) * p.γ * (1 - p.γ) + 2*D_const*C_const* (p11(p.α, p.β) - p.γ^2)) + (p11(p.α, p.β) - p.γ^2)*( 2*D_const*C_const * p.γ*(1 - p.γ) + (D_const^2 + C_const^2) * (p11(p.α, p.β) - p.γ^2))) / ((D_const^2 - C_const^2) *p.γ*(1 - p.γ))^2
            end
            
            if abs(r_fin - r_val[k]) < bound_r /(resolution_factor * resolution)
                println("R = ", r_val[k])
                if r_val[k] != 0.0      
                    m1 = m1_old + h * (- m1_old + nullcline1(p, g, n_temp, h11_val, h10_val, h01_val, h00_val))/tau_m
                    m2 = m2_old + h * (- m2_old + nullcline2(p, g, n_temp, h11_val, h10_val, h01_val, h00_val))/tau_m
                    r = r_val[k]
                    continue
                #else 
                println("after computing the nullclines")
                """
                    n0 = NoNoise()
                    h11_val = h11(p, g, m1_old, m2_old, I1, I2)
                    h10_val = h10(p, g, m1_old, m2_old, I1, I2)
                    h01_val = h01(p, g, m1_old, m2_old, I1, I2)
                    h00_val = h00(p, g, m1_old, m2_old, I1, I2)       
                    m1 = m1_old + h * (- m1_old + nullcline1(p, g, n0, h11_val, h10_val, h01_val, h00_val))/tau_m
                    m2 = m2_old + h * (- m2_old + nullcline2(p, g, n0, h11_val, h10_val, h01_val, h00_val))/tau_m
                    r = 0.
                """
                end
                #continue
            end
        end
    else 
        n0 = NoNoise()
        h11_val = h11(p, g, m1_old, m2_old, I1, I2)
        h10_val = h10(p, g, m1_old, m2_old, I1, I2)
        h01_val = h01(p, g, m1_old, m2_old, I1, I2)
        h00_val = h00(p, g, m1_old, m2_old, I1, I2)       
        m1 = m1_old + h * (- m1_old + nullcline1(p, g, n0, h11_val, h10_val, h01_val, h00_val))/tau_m
        m2 = m2_old + h * (- m2_old + nullcline2(p, g, n0, h11_val, h10_val, h01_val, h00_val))/tau_m
        r = 0.
    end
    return  m1, m2, r
end
### NOTE FOR A MORE POLISHED VERSION OF THE CODE:  Uniform this function with the adaptive version and make the while loop optional!!!
function run_dynamics(; rm = 1, A = 1, h0 = 0, γ = 0.1 , b = 1000, I2_val = 0.1, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load = 0, C_hat = 0. , final_time = 15, t_onset = 0.5, t_offset = 5.5, h = 0.01, tau_m = 1, model, corr_noise = false)
    X_p = 0.
    Y_p = 0. # we initialize the state of the network in the first pattern
    r = 0.
    I1 = 0.
    I2 = 0.
    t_onset2 = 8.
    t_offset2 = 13.
    p = NetworkParameters(γ = γ , A = A, C_hat = C_hat)
    g = Gain( rm = rm, b = b, h0 = h0)
    result = @distributed vcat for i in 0:round(Int,final_time/h)  # to be modified not distributed?
    tmp = DataFrame(time = [], x_p = [], y_p = [], r_p = [],  stab = [])
    if i == 0
        push!(tmp, [0, X_p, Y_p, r,  2])
    else
        running_time = i  * h
        println("running_time = ", running_time)
        if running_time >= t_onset && running_time < t_offset
            I1 = 0.3
        elseif running_time > t_offset
            I1 = 0.
        end
        if running_time > t_onset2 && running_time < t_offset2
            println("in the if statement")
            I2 = I2_val
        elseif running_time >= t_offset2
            I2 = 0.
        end
        noise = NoNoise()
        if r > 0 
            noise = Noise(load = load, r = r, A = A)
        end
        ### NOTE FOR A MORE POLISHED VERSION OF THE CODE: It is not very useful to have the stability here if the update doesn't converge to a fixed points - it can be taken out
        for j in 1: length(X_p)
            eigs = real( eigenvalues(X_p[j], Y_p[j], I1, I2, p, g, noise))  
            if eigs[1] > 0 && eigs[2] > 0
                push!(tmp, [running_time, X_p[j], Y_p[j], r,  0])
                #println("unstable")
            elseif (eigs[1] > 0 && eigs[2] < 0) || (eigs[1] < 0 && eigs[2] > 0)
                push!(tmp, [running_time, X_p[j], Y_p[j], r,   1])
                #println("saddle")
            elseif eigs[1] < 0 && eigs[2] < 0
                #println("stable")
                push!(tmp, [running_time, X_p[j], Y_p[j], r,  2])
            end
        end
        println("I1 = ", I1, "      I2 = ", I2)
        X_p, Y_p, r = update_state(p, g, X_p, Y_p, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, running_time, h, t_onset, t_offset, tau_m, model, corr_noise)
    end
    
    tmp
    end
    CSV.write("Output_2pop/MF_vs_time_gamma$(γ)_time_step$(h)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_Chat$(C_hat).dat", result,  delim = ' ')
    println("I1 = ",I1)
    generate_PPs(p, g, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, model, corr_noise)
    println("first PP generated")
    I1 = 1.
    generate_PPs(p, g, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, model, corr_noise)
end

function generate_PPs(p, g, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, model, corr_noise)
    M11, M21, M12, M22, R1 = get_nullclines(p, g, I1, I2, bound_low, bound_up, bound_r, resolution, resolution_factor, load, model, corr_noise)
    X_p, Y_p, R_p = get_fixed_points(M11, M21, M21, M11, R1, bound_up, bound_low, resolution, resolution_factor)  
    stab = get_fp_stability(X_p, Y_p, R_p, I1, I2, p, g, load)
    open("Output_2pop/nullcline1_gamma$(p.γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(g.rm)_A$(p.A)_h0$(g.h0)_b$(g.b)_load$(load)_C_hat$(p.C_hat)_I1$(I1).dat","w") do null1
        open("Output_2pop/nullcline2_gamma$(p.γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(g.rm)_A$(p.A)_h0$(g.h0)_b$(g.b)_load$(load)_C_hat$(p.C_hat)_I1$(I1).dat","w") do null2
            for j in 1:length(M11)
                write(null1, "$(M11[j])   $(M21[j])   \n")
            end
            for j in 1:length(M22)
                write(null2, "$(M12[j])   $(M22[j])   \n")
            end
        end 
    end
    open("Output_2pop/fp_gamma$(p.γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(g.rm)_A$(p.A)_h0$(g.h0)_b$(g.b)_load$(load)_C_hat$(p.C_hat)_I1$(I1).dat","w") do fp
        for i in 1:length(X_p)
            write(fp,"$(X_p[i])  $(Y_p[i])   $(R_p[i])   $(stab[i])\n")
        end
    end
end

function compute_I_C_curve(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, tau_m = 1, h = 0.1, size_I, size_C, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load, model = "MeanField", corr_noise = false)
    #g = Gain( rm = rm, b = b, h0 = h0)
    #I2_fin = nan
    I2_vect = Float64[]
    C_vect = Float64[]
    #open("Output_2pop/generate_I_C_curve_gamma$(γ)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load).dat","w") do file
        c_hat_range = range( 0., stop = h0/rm + h0/(10*rm)  , length = size_C )  
        I2_range = range( 0., stop = h0/rm + h0/(10*rm)  , length = size_I )  
        for C in c_hat_range
            #p = NetworkParameters(γ = γ , A = A, C_hat = C)
            for I2 in I2_range
                run_dynamics( rm = rm, A = A, h0 = h0, γ = γ , b = b, I2_val = I2, bound_low = bound_low, bound_up = bound_up, bound_r = bound_r, resolution = resolution, resolution_factor = resolution_factor, load = load, C_hat = C , final_time = 15, t_onset = 0.5, t_offset = 5.5, h = 0.1, tau_m = 1, model = model, corr_noise = corr_noise)
                (data_cells, header_cells) = readdlm("Output_2pop/MF_vs_time_gamma$(γ)_time_step$(h)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_Chat$(C).dat", header = true)
                #println(data_cells[1])
                time = data_cells[:, 1]
                m1 = data_cells[:, 2]
                m2 = data_cells[:, 3]
                #println(m2)
                r_p = data_cells[:, 4]
                stab = data_cells[:, 5]
                if m2[length(m2)] > 0.9 #&& (I2_vect[end]-I2)>0.5 #&& length(I2_vect) > 0
                    #return C, I2
                    push!(I2_vect, I2)
                    push!(C_vect, C)
                    break
                end
            end
        end
    #end
    return C_vect, I2_vect
end

function generate_I_C_curve(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, tau_m = 1, h = 0.1, size_I, size_C, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load, model = "MeanField", corr_noise = false)
    open("Output_2pop/generate_I_C_curve_gamma$(γ)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load).dat","w") do file
        C, I2 = compute_I_C_curve( rm = rm, A = A, h0 = h0, γ = γ , b = b, tau_m = tau_m, h = h, size_I = size_I, size_C = size_C, bound_low = bound_low, bound_up = bound_up, bound_r = bound_r, resolution = resolution, resolution_factor = resolution_factor, load = load, model = model, corr_noise = corr_noise)
        for i in 1:length(C)
            write(file, "$(C[i])   $(I2[i])\n")
        end
    end
end
end # end of module TwoPopulations

module ThreePopulations
import ..AttractorNetwork: NetworkParameters, get_α_β, Heaviside, Gain, dgain_dx, gain_squared, Noise, NoNoise, p1, p11, p10, p00
using PolynomialRoots
using HCubature
using SpecialFunctions
using Roots
using NLsolve
using Distributed, DataFrames, CSV
using LinearAlgebra

struct Results
    M1 :: Vector{Float64}  
    M2 :: Vector{Float64}  
    M3 :: Vector{Float64}  
    R :: Vector{Float64}  
end
Results() = Results([], [], [], [])

# Joint probabilities for correlated patterns
p111(α, β) = α * β^3 + (1 - α) * (1 - β)^3
p111(p) = p111(p.α, p.β)
p110( α, β) = α * β^2 * (1 - β) + (1 - α) * β * (1 - β)^2
p110(p) = p110(p.α, p.β)
p100( α, β) = α * β * (1 - β)^2 + (1 - α) * β^2 * (1 - β)
p100(p) = p100(p.α, p.β)
p000(α, β) = α * (1 - β)^3 + (1 - α) * β^3
p000(p) = p000(p.α, p.β)

# Define the most used 4 inputs, no noise case
h111(p, g, m1, m2, m3) = p.A * g.rm * (1 - p.γ) * (m1 + m2 + m3) 
h000(p, g, m1, m2, m3) = -p.γ * p.A * g.rm * (m1 + m2 + m3)
h110(p, g, m1, m2, m3) = p.A * g.rm * (1 - p.γ) * (m1 + m2) - p.γ * p.A * g.rm * m3
h001(p, g, m1, m2, m3) = p.A * g.rm * (1 - p.γ) * m3 - p.γ * p.A * g.rm * (m1 + m2)
h101(p, g, m1, m2, m3) = h110(p, g, m1, m3, m2)
h011(p, g, m1, m2, m3) = h110(p, g, m2, m3, m1)
h010(p, g, m1, m2, m3) = h001(p, g, m1, m3, m2)
h100(p, g, m1, m2, m3) = h001(p, g, m1, m3, m1)

# Here we implement the 4 entries of the Jacobian matrix 
function J11(p, g, m1, m2, m3) 
    - 1 + p.A / ( p.γ * (1 - p.γ)) * ((1 - p.γ)^2 * p111(p) * dgain_dx(g, h111(p, g, m1, m2, m3)) + 
        (1 - p.γ)^2 * p110(p) * dgain_dx(g, h110(p, g, m1, m2, m3)) +  
        (1 - p.γ)^2 * p110(p) * dgain_dx(g, h101(p, g, m1, m2, m3)) +
        (1 - p.γ)^2 * p100(p) * dgain_dx(g, h100(p, g, m1, m2, m3)) +
        p.γ^2 * p110(p) * dgain_dx(g, h011(p, g, m1, m2, m3)) +
        p.γ^2 * p100(p) * dgain_dx(g, h001(p, g, m1, m2, m3)) +
        p.γ^2 * p100(p) * dgain_dx(g, h010(p, g, m1, m2, m3)) +
        p.γ^2 * p000(p) * dgain_dx(g, h000(p, g, m1, m2, m3)))
end
function J12(p, g, m1, m2, m3) 
    p.A / ( p.γ * (1 - p.γ)) * ((1 - p.γ)^2 * p111(p) * dgain_dx(g, h111(p, g, m1, m2, m3)) + 
        (1 - p.γ)^2 * p110(p) * dgain_dx(g, h110(p, g, m1, m2, m3)) -  
        p.γ * (1 - p.γ) * p110(p) * dgain_dx(g, h101(p, g, m1, m2, m3)) -
        p.γ * (1 - p.γ) * p100(p) * dgain_dx(g, h100(p, g, m1, m2, m3)) -
        p.γ * (1 - p.γ) * p110(p) * dgain_dx(g, h011(p, g, m1, m2, m3)) +
        p.γ^2 * p100(p) * dgain_dx(g, h001(p, g, m1, m2, m3)) -
        p.γ * (1 - p.γ) * p100(p) * dgain_dx(g, h010(p, g, m1, m2, m3)) +
        p.γ^2 * p000(p) * dgain_dx(g, h000(p, g, m1, m2, m3)))
end
function J13(p, g, m1, m2, m3) 
    p.A / ( p.γ * (1 - p.γ)) * ((1 - p.γ)^2 * p111(p) * dgain_dx(g, h111(p, g, m1, m2, m3)) - 
        p.γ * (1 - p.γ) * p110(p) * dgain_dx(g, h110(p, g, m1, m2, m3)) +  
        (1 - p.γ)^2 * p110(p) * dgain_dx(g, h101(p, g, m1, m2, m3)) -
        p.γ * (1 - p.γ) * p100(p) * dgain_dx(g, h100(p, g, m1, m2, m3)) -
        p.γ * (1 - p.γ) * p110(p) * dgain_dx(g, h011(p, g, m1, m2, m3)) -
        p.γ * (1 - p.γ) * p100(p) * dgain_dx(g, h001(p, g, m1, m2, m3)) +
        (1 - p.γ)^2 * p100(p) * dgain_dx(g, h010(p, g, m1, m2, m3)) +
        p.γ^2 * p000(p) * dgain_dx(g, h000(p, g, m1, m2, m3)))
end
J22(p, g, m1, m2, m3) = J11(p, g, m2, m1, m3)
J33(p, g, m1, m2, m3) = J11(p, g, m3, m2, m1)
J23(p, g, m1, m2, m3) = J12(p, g, m2, m3, m1)

function compute_eigenvalues(p, g, m1, m2, m3)
    J = [J11(p, g, m1, m2, m3)  J12(p, g, m1, m2, m3)   J13(p, g, m1, m2, m3); 
        J12(p, g, m1, m2, m3)  J22(p, g, m1, m2, m3)   J23(p, g, m1, m2, m3);
        J13(p, g, m1, m2, m3)  J23(p, g, m1, m2, m3)   J33(p, g, m1, m2, m3)]
    return eigvals(J)
end 

# System's variables: m1, m2, p, q
function nullcline1(p, g, m1, m2, m3)
    1 / (g.rm * p.γ * (1 - p.γ)) * ((1 - p.γ) * p111(p) * g(h111(p, g, m1, m2, m3)) + 
        (1 - p.γ) * p110(p) * g(h110(p, g, m1, m2, m3)) +  
        (1 - p.γ) * p110(p) * g(h101(p, g, m1, m2, m3)) +
        (1 - p.γ) * p100(p) * g(h100(p, g, m1, m2, m3)) -
        p.γ * p110(p) * g(h011(p, g, m1, m2, m3)) -
        p.γ * p100(p) * g(h001(p, g, m1, m2, m3)) -
        p.γ * p100(p) * g(h010(p, g, m1, m2, m3)) -
        p.γ * p000(p) * g(h000(p, g, m1, m2, m3)))
end

function nullcline1(p, g::Gain, n::Noise, m1, m2, m3)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi) * g.rm * p.γ * (1 - p.γ)) * (
    (1 - p.γ) * p111(p) * g(h111(p, g, m1, m2, m3)+ n.σ*x[1]) + 
    (1 - p.γ) * p110(p) * g(h110(p, g, m1, m2, m3)+ n.σ*x[1]) +  
    (1 - p.γ) * p110(p) * g(h101(p, g, m1, m2, m3)+ n.σ*x[1]) +
    (1 - p.γ) * p100(p) * g(h100(p, g, m1, m2, m3)+ n.σ*x[1]) -
    p.γ * p110(p) * g(h011(p, g, m1, m2, m3)+ n.σ*x[1]) -
    p.γ * p100(p) * g(h001(p, g, m1, m2, m3)+ n.σ*x[1]) -
    p.γ * p100(p) * g(h010(p, g, m1, m2, m3)+ n.σ*x[1]) -
    p.γ * p000(p) * g(h000(p, g, m1, m2, m3)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

denominator(p, n) = sqrt(2)* n.σ
function nullcline1(p, g::Heaviside, n::Noise, m1, m2, m3)
    result = 1. / (p.γ * (1 - p.γ)) * 0.5 * (
        (1 - p.γ) * p111(p) * erfc((g.h0 - h111(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        (1 - p.γ) * p110(p) * erfc((g.h0 - h110(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +  
        (1 - p.γ) * p110(p) * erfc((g.h0 - h101(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        (1 - p.γ) * p100(p) * erfc((g.h0 - h100(p, g, m1, m2 , m3, n) ) / denominator(p, n)) -
        p.γ * p110(p) * erfc((g.h0 - h011(p, g, m1, m2 , m3, n) ) / denominator(p, n)) -
        p.γ * p100(p) * erfc((g.h0 - h001(p, g, m1, m2 , m3, n) ) / denominator(p, n)) -
        p.γ * p100(p) * erfc((g.h0 - h010(p, g, m1, m2 , m3, n) ) / denominator(p, n)) -
        p.γ * p000(p) * erfc((g.h0 - h000(p, g, m1, m2 , m3, n) ) / denominator(p, n)))
    return result
end
nullcline2(p, g, n,  m1, m2, m3) = nullcline1(p, g, n,  m2, m1, m3) #exchange m1 with m2
nullcline3(p, g, n,  m1, m2, m3) = nullcline1(p, g, n,  m3, m2, m1)
nullcline2(p, g, m1, m2, m3) = nullcline1(p, g, m2, m1, m3)
nullcline3(p, g, m1, m2, m3) = nullcline1(p, g, m3, m2, m1)

function compute_q(p, g::Gain, n::Noise, m1, m2, m3) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p111(p) * dgain_dx(g, h111(p, g, m1, m2, m3)+ n.σ*x[1]) + 
        p110(p) * dgain_dx(g, h110(p, g, m1, m2, m3)+ n.σ*x[1]) +  
        p110(p) * dgain_dx(g, h101(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p100(p) * dgain_dx(g, h100(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p110(p) * dgain_dx(g, h011(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p100(p) * dgain_dx(g, h001(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p100(p) * dgain_dx(g, h010(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p000(p) * dgain_dx(g, h000(p, g, m1, m2, m3)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_q(p, g::Heaviside, n::Noise, m1, m2, m3)
    result = g.rm/sqrt(2*pi) * (p111(p) * exp(- ((g.h0 - h111(p, g, m1, m2, m3))/denominator(p, n))^2)  +
        p110(p) * exp(-((g.h0 - h110(p, g, m1, m2 , m3, n) ) / denominator(p, n))^2) +  
        p110(p) * exp(-((g.h0 - h101(p, g, m1, m2 , m3, n) ) / denominator(p, n))^2) +
        p100(p) * exp(-((g.h0 - h100(p, g, m1, m2 , m3, n) ) / denominator(p, n))^2) +
        p110(p) * exp(-((g.h0 - h011(p, g, m1, m2 , m3, n) ) / denominator(p, n))^2) +
        p100(p) * exp(-((g.h0 - h001(p, g, m1, m2 , m3, n) ) / denominator(p, n))^2) +
        p100(p) * exp(-((g.h0 - h010(p, g, m1, m2 , m3, n) ) / denominator(p, n))^2) +
        p000(p) * exp(-((g.h0 - h000(p, g, m1, m2 , m3, n) ) / denominator(p, n))^2))
    return result
end 

function compute_p(p, g::Gain, n::Noise, m1, m2, m3) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p111(p) * gain_squared(g, h111(p, g, m1, m2, m3)+ n.σ*x[1]) + 
        p110(p) * gain_squared(g, h110(p, g, m1, m2, m3)+ n.σ*x[1]) +  
        p110(p) * gain_squared(g, h101(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p100(p) * gain_squared(g, h100(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p110(p) * gain_squared(g, h011(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p100(p) * gain_squared(g, h001(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p100(p) * gain_squared(g, h010(p, g, m1, m2, m3)+ n.σ*x[1]) +
        p000(p) * gain_squared(g, h000(p, g, m1, m2, m3)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_p(p, g::Heaviside, n::Noise, m1, m2, m3) 
    result = g.rm^2 * 0.5 * (
        p111(p) * erfc((g.h0 - h111(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        p110(p) * erfc((g.h0 - h110(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +  
        p110(p) * erfc((g.h0 - h101(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        p100(p) * erfc((g.h0 - h100(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        p110(p) * erfc((g.h0 - h011(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        p100(p) * erfc((g.h0 - h001(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        p100(p) * erfc((g.h0 - h010(p, g, m1, m2 , m3, n) ) / denominator(p, n)) +
        p000(p) * erfc((g.h0 - h000(p, g, m1, m2 , m3, n) ) / denominator(p, n)))
    return result
end

function push_to_fixed_point!(results, m1, m2, m3, m_val1, m_val2, m_val3, bound_up, bound_low, resolution_factor, resolution, r)
    if abs(m1-m_val1)<(bound_up-bound_low)/(resolution_factor*resolution) && 
        abs(m2-m_val2)<(bound_up-bound_low)/(resolution_factor*resolution) && 
        abs(m3-m_val3)<(bound_up-bound_low)/(resolution_factor*resolution)
        push!(results.M1, m_val1)
        push!(results.M2, m_val2)
        push!(results.M3, m_val3)
        push!(results.R, r)
    end
    """
    #push!(m1_temp, m1 - m_val1)  
    #push!(m2_temp, m2 - m_val2)
    #push!(m3_temp, m2 - m_val3)
    temp_m1 = Float64[]
    temp_m2 = Float64[]
    temp_m3 = Float64[]
        
    if length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0  
        push!(temp_m1, m_val1)
    end
    if length(m2_temp) > 1 &&  m2_temp[end]*m2_temp[end-1] < 0
        push!(temp_m2, m_val2)
    end
    if length(m3_temp) > 1 &&  m3_temp[end]*m3_temp[end-1] < 0
        push!(temp_m3, m_val3)
    end
    """

end

function get_fixed_points(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
    m_val = range( bound_low, stop= bound_up, length = resolution)
    r_val = range( 0, stop= bound_r, length = resolution)
    @show load
    results = Results()
    """
    m1 = 0
    m2 = 0
    m3 = 0
    m3_temp = Float64[]
    """
    for k in 1 : resolution
        #m2_temp = Float64[]
        for j in 1 : resolution
            #m1_temp = Float64[]
            for i in 1 : resolution
                if load != 0.
                    for l in 1 : resolution
                        n_temp = Noise(load = load, r = r_val[l], A = p.A)
                        q_func = compute_q(p, g, n_temp, m_val[i], m_val[j], m_val[k])
                        p_func = compute_p(p, g, n_temp, m_val[i], m_val[j], m_val[k]) 
                        D_const = (1 - p.A * q_func)
                        r_fin = p_func /(D_const^2)
                        
                        if abs(r_fin - r_val[l]) < bound_r /(resolution_factor * resolution)
                            if r_val[l] != 0.0            
                                m1 = nullcline1(p, g, n_temp, m_val[i], m_val[j], m_val[k])
                                m2 = nullcline2(p, g, n_temp, m_val[i], m_val[j], m_val[k])
                                m3 = nullcline3(p, g, n_temp, m_val[i], m_val[j], m_val[k])
                                push_to_fixed_point!(results, m1, m2, m3, m_val[i], m_val[j], m_val[k],  bound_up, bound_low, resolution_factor, resolution, r_val[l])
                                
                            else 
                                m1 = nullcline1(p, g, m_val[i], m_val[j], m_val[k])
                                m2 = nullcline2(p, g, m_val[i], m_val[j], m_val[k])
                                m3 = nullcline3(p, g, m_val[i], m_val[j], m_val[k])
                                push_to_fixed_point!(results, m1, m2, m3, m_val[i], m_val[j], m_val[k], bound_up, bound_low, resolution_factor, resolution, 0.)
                            end
                            continue
                        end
                    end
                else 
                    m1 = nullcline1(p, g, m_val[i], m_val[j], m_val[k])
                    m2 = nullcline2(p, g, m_val[i], m_val[j], m_val[k])
                    m3 = nullcline3(p, g, m_val[i], m_val[j], m_val[k])
                    push_to_fixed_point!(results, m1, m2, m3, m_val[i], m_val[j], m_val[k], bound_up, bound_low, resolution_factor, resolution, 0.)
                end
            end
        end
    end
    return  results.M1, results.M2 , results.M3, results.R  
end 

function generate_bifurcation_diagram(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution, resolution_factor = 1, size, load)
    g = Gain( rm = rm, b = b, h0 = h0 )
    c_hat_range = range( 0., stop = h0/rm , length = size )

    result = @distributed vcat for i in 1:size
        tmp = DataFrame(C = [], x_p = [], r_p = [], stab = [])
        
        start = time()
        p = NetworkParameters(γ = γ , A = A, C_hat = c_hat_range[i])
        println( "C_hat = ", p.C_hat, "   h0 = ", h0)
        
        M1, M2, M3, R = get_fixed_points(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
        
        for j in 1: length(M1)
            eigs = real(compute_eigenvalues(p, g, M1[j], M2[j], M3[j]))
            if eigs[1] > 0 && eigs[2] > 0 && eigs[3] > 0
                push!(tmp, [p.C_hat, M1[j], R[j],  0])
            elseif eigs[1] < 0 && eigs[2] < 0 && eigs[3] < 0
                push!(tmp, [p.C_hat, M1[j], R[j],  2])
            else 
                push!(tmp, [p.C_hat, M1[j], R[j],  1])
            end
        end
        
        elapsed = time() - start
        println("Simulation of point ",  i ," of ", size, " took ", elapsed, " seconds")
        tmp
    end
    CSV.write("Output_3pop/Bifurcation_plot_vs_C_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load).dat", result)
end

end # end of module ThreePopulations

module FourPopulations
import ..AttractorNetwork: NetworkParameters, get_α_β, Heaviside, Gain, dgain_dx, gain_squared, Noise, NoNoise, p1, p11, p10, p00
using PolynomialRoots
using HCubature
using SpecialFunctions
using Roots
using NLsolve
using Distributed, DataFrames, CSV
using LinearAlgebra

struct Results
    M1 :: Vector{Float64}  
    M2 :: Vector{Float64}  
    M3 :: Vector{Float64}  
    M4 :: Vector{Float64}  
    R :: Vector{Float64}  
end
Results() = Results([], [], [], [], [])

# Joint probabilities for correlated patterns
p1111(α, β) = α * β^4 + (1 - α) * (1 - β)^4
p1111(p) = p1111(p.α, p.β)
p1110( α, β) = α * β^3 * (1 - β) + (1 - α) * β * (1 - β)^3
p1110(p) = p1110(p.α, p.β)
p1100( α, β) = α * β^2 * (1 - β)^2 + (1 - α) * β^2 * (1 - β)^2
p1100(p) = p1100(p.α, p.β)
p1000( α, β) = α * β * (1 - β)^3 + (1 - α) * β^3 * (1 - β)
p1000(p) = p1000(p.α, p.β)
p0000(α, β) = α * (1 - β)^4 + (1 - α) * β^4
p0000(p) = p0000(p.α, p.β)

# Define the most used 4 inputs, no noise case
h1111(p, g, m1, m2, m3 , m4) = p.A * g.rm * (1 - p.γ) * (m1 + m2 + m3 + m4) 
h0000(p, g, m1, m2, m3, m4) = - p.γ * p.A * g.rm * (m1 + m2 + m3 + m4)
h1110(p, g, m1, m2, m3, m4) = p.A * g.rm * (1 - p.γ) * (m1 + m2 +m3) - p.γ * p.A * g.rm * m4
h1101(p, g, m1, m2, m3, m4) = h1110(p, g, m1, m2, m4, m3)
h1011(p, g, m1, m2, m3, m4) = h1110(p, g, m1, m4, m3, m2)
h0111(p, g, m1, m2, m3, m4) = h1110(p, g, m4, m2, m3, m1)
h1100(p, g, m1, m2, m3, m4) = p.A * g.rm * (1 - p.γ) * (m1 + m2 ) - p.γ * p.A * g.rm * (m3 + m4)
h1010(p, g, m1, m2, m3, m4) = h1100(p, g, m1, m3, m2, m4)
h0101(p, g, m1, m2, m3, m4) = h1100(p, g, m4, m2, m3, m1)
h1001(p, g, m1, m2, m3, m4) = h1100(p, g, m1, m4, m3, m2)
h0110(p, g, m1, m2, m3, m4) = h1100(p, g, m3, m2, m1, m4)
h0011(p, g, m1, m2, m3, m4) = h1100(p, g, m3, m4, m1, m2)
h1000(p, g, m1, m2, m3, m4) = p.A * g.rm * (1 - p.γ) * (m1 ) - p.γ * p.A * g.rm * (m2 + m3 + m4)
h0100(p, g, m1, m2, m3, m4) = h1000(p, g, m2, m1, m3, m4)
h0010(p, g, m1, m2, m3, m4) = h1000(p, g, m3, m2, m1, m4)
h0001(p, g, m1, m2, m3, m4) = h1000(p, g, m4, m2, m3, m1)

# Here we implement the 4 entries of the Jacobian matrix 
function J11(p, g, m1, m2, m3, m4) 
    - 1 + p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p1111(p) * dgain_dx(g,h1111(p, g, m1, m2, m3, m4)) + 
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g, h1110(p, g, m1, m2, m3, m4)) +  
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g, h1101(p, g, m1, m2, m3, m4)) +
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g, h1011(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1110(p) * dgain_dx(g, h0111(p, g, m1, m2, m3, m4)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g, h1100(p, g, m1, m2, m3, m4)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g, h1010(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1100(p) * dgain_dx(g, h0110(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1100(p) * dgain_dx(g, h0011(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1100(p) * dgain_dx(g, h0101(p, g, m1, m2, m3, m4)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g, h1001(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1000(p) * dgain_dx(g, h0001(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1000(p) * dgain_dx(g, h0010(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1000(p) * dgain_dx(g, h0100(p, g, m1, m2, m3, m4)) +
        (1 - p.γ)^2 * p1000(p) * dgain_dx(g, h1000(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p0000(p) * dgain_dx(g, h0000(p, g, m1, m2, m3, m4)))
end
function J12(p, g, m1, m2, m3, m4) 
    p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p1111(p) * dgain_dx(g,h1111(p, g, m1, m2, m3, m4)) + 
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g, h1110(p, g, m1, m2, m3, m4)) +  
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g, h1101(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1110(p) * dgain_dx(g, h1011(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1110(p) * dgain_dx(g, h0111(p, g, m1, m2, m3, m4)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g, h1100(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g, h1010(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g, h0110(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1100(p) * dgain_dx(g, h0011(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g, h0101(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g, h1001(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1000(p) * dgain_dx(g, h0001(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p1000(p) * dgain_dx(g, h0010(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1000(p) * dgain_dx(g, h0100(p, g, m1, m2, m3, m4)) -
        p.γ * (1 - p.γ) * p1000(p) * dgain_dx(g, h1000(p, g, m1, m2, m3, m4)) +
        p.γ^2 * p0000(p) * dgain_dx(g, h0000(p, g, m1, m2, m3, m4)))
end
J13(p, g, m1, m2, m3, m4) = J12(p, g, m1, m3, m2, m4) 
J14(p, g, m1, m2, m3, m4) = J12(p, g, m1, m4, m3, m2) 
J22(p, g, m1, m2, m3, m4) = J11(p, g, m2, m1, m3, m4)
J33(p, g, m1, m2, m3, m4) = J11(p, g, m3, m2, m1, m4)
J44(p, g, m1, m2, m3, m4) = J11(p, g, m4, m2, m3, m1)
J23(p, g, m1, m2, m3, m4) = J12(p, g, m2, m3, m1, m4)
J24(p, g, m1, m2, m3, m4) = J12(p, g, m2, m4, m3, m1)   
J34(p, g, m1, m2, m3, m4) = J12(p, g, m3, m4, m1, m2) 

function compute_eigenvalues(p, g, m1, m2, m3, m4)
    J = [J11(p, g, m1, m2, m3, m4)  J12(p, g, m1, m2, m3, m4)   J13(p, g, m1, m2, m3, m4)   J14(p, g, m1, m2, m3, m4); 
        J12(p, g, m1, m2, m3, m4)  J22(p, g, m1, m2, m3, m4)   J23(p, g, m1, m2, m3, m4)    J24(p, g, m1, m2, m3, m4);
        J13(p, g, m1, m2, m3, m4)  J23(p, g, m1, m2, m3, m4)   J33(p, g, m1, m2, m3, m4)    J34(p, g, m1, m2, m3, m4);
        J14(p, g, m1, m2, m3, m4)  J24(p, g, m1, m2, m3, m4)   J34(p, g, m1, m2, m3, m4)    J44(p, g, m1, m2, m3, m4)]
    return eigvals(J)
end 

# System's variables: m1, m2, p, q
function nullcline1(p, g, m1, m2, m3, m4)
    1 / (g.rm * p.γ * (1 - p.γ)) * (
        (1 - p.γ) * p1111(p) * g(h1111(p, g, m1, m2, m3, m4)) + 
        (1 - p.γ) * p1110(p) * g(h1110(p, g, m1, m2, m3, m4)) +  
        (1 - p.γ) * p1110(p) * g(h1101(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1110(p) * g(h1011(p, g, m1, m2, m3, m4)) -
        p.γ * p1110(p) * g(h0111(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1100(p) * g(h1100(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1100(p) * g(h1010(p, g, m1, m2, m3, m4)) -
        p.γ * p1100(p) * g(h0110(p, g, m1, m2, m3, m4)) -
        p.γ * p1100(p) * g(h0011(p, g, m1, m2, m3, m4)) -
        p.γ * p1100(p) * g(h0101(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1100(p) * g(h1001(p, g, m1, m2, m3, m4)) -
        p.γ * p1000(p) * g(h0001(p, g, m1, m2, m3, m4)) -
        p.γ * p1000(p) * g(h0010(p, g, m1, m2, m3, m4)) -
        p.γ * p1000(p) * g(h0100(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1000(p) * g(h1000(p, g, m1, m2, m3, m4)) -
        p.γ * p0000(p) * g(h0000(p, g, m1, m2, m3, m4)))
end

function nullcline1(p, g::Gain, n::Noise, m1, m2, m3, m4)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi) * g.rm * p.γ * (1 - p.γ)) * (
        (1 - p.γ) * p1111(p) * g(h1111(p, g, m1, m2, m3, m4)) + 
        (1 - p.γ) * p1110(p) * g(h1110(p, g, m1, m2, m3, m4)) +  
        (1 - p.γ) * p1110(p) * g(h1101(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1110(p) * g(h1011(p, g, m1, m2, m3, m4)) -
        p.γ * p1110(p) * g(h0111(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1100(p) * g(h1100(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1100(p) * g(h1010(p, g, m1, m2, m3, m4)) -
        p.γ * p1100(p) * g(h0110(p, g, m1, m2, m3, m4)) -
        p.γ * p1100(p) * g(h0011(p, g, m1, m2, m3, m4)) -
        p.γ * p1100(p) * g(h0101(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1100(p) * g(h1001(p, g, m1, m2, m3, m4)) -
        p.γ * p1000(p) * g(h0001(p, g, m1, m2, m3, m4)) -
        p.γ * p1000(p) * g(h0010(p, g, m1, m2, m3, m4)) -
        p.γ * p1000(p) * g(h0100(p, g, m1, m2, m3, m4)) +
        (1 - p.γ) * p1000(p) * g(h1000(p, g, m1, m2, m3, m4)) -
        p.γ * p0000(p) * g(h0000(p, g, m1, m2, m3, m4)))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

denominator(p, n) = sqrt(2)* n.σ
function nullcline1(p, g::Heaviside, n::Noise, m1, m2, m3)
    1 / (p.γ * (1 - p.γ)) * 0.5 * (
        (1 - p.γ) * p1111(p) * erfc((g.h0 - h1111(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) + 
        (1 - p.γ) * p1110(p) * erfc((g.h0 - h1110(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +  
        (1 - p.γ) * p1110(p) * erfc((g.h0 - h1101(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        (1 - p.γ) * p1110(p) * erfc((g.h0 - h1011(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) -
        p.γ * p1110(p) * erfc((g.h0 - h0111(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        (1 - p.γ) * p1100(p) * erfc((g.h0 - h1100(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        (1 - p.γ) * p1100(p) * erfc((g.h0 - h1010(p, g, m1, m2 , m3, m4, n)  )/ denominator(p, n)) -
        p.γ * p1100(p) * erfc((g.h0 - h0110(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) -
        p.γ * p1100(p) * erfc((g.h0 - h0011(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) -
        p.γ * p1100(p) * erfc((g.h0 - h0101(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        (1 - p.γ) * p1100(p) * erfc((g.h0 - h1001(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) -
        p.γ * p1000(p) * erfc((g.h0 - h0001(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) -
        p.γ * p1000(p) * erfc((g.h0 - h0010(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) -
        p.γ * p1000(p) * erfc((g.h0 - h0100(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        (1 - p.γ) * p1000(p) * erfc((g.h0 - h1000(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) -
        p.γ * p0000(p) * erfc((g.h0 - h0000(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)))
    return result
end
nullcline2(p, g, n,  m1, m2, m3, m4) = nullcline1(p, g, n,  m2, m1, m3, m4) #exchange m1 with m2
nullcline3(p, g, n,  m1, m2, m3, m4) = nullcline1(p, g, n,  m3, m2, m1, m4)
nullcline4(p, g, n,  m1, m2, m3, m4) = nullcline1(p, g, n,  m4, m2, m3, m1)
nullcline2(p, g, m1, m2, m3, m4) = nullcline1(p, g, m2, m1, m3, m4)
nullcline3(p, g, m1, m2, m3, m4) = nullcline1(p, g, m3, m2, m1, m4)
nullcline4(p, g, m1, m2, m3, m4) = nullcline1(p, g, m4, m2, m3, m1)

function compute_q(p, g::Gain, n::Noise, m1, m2, m3, m4) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p1111(p) * dgain_dx(g, h1111(p, g, m1, m2, m3, m4)+ n.σ*x[1]) + 
        p1110(p) * dgain_dx(g, h1110(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +  
        p1110(p) * dgain_dx(g, h1101(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1110(p) * dgain_dx(g, h1011(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1110(p) * dgain_dx(g, h0111(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g, h1100(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g, h1010(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g, h0110(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g, h0011(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g, h0101(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g, h1001(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g, h0001(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g, h0010(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g, h0100(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g, h1000(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p0000(p) * dgain_dx(g, h0000(p, g, m1, m2, m3, m4)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_q(p, g::Heaviside, n::Noise, m1, m2, m3, m4)
    result = g.rm/sqrt(2*pi) * (
        p1111(p) * exp(- ((g.h0 - h1111(p, g, m1, m2, m3, m4)) / denominator(p, n))^2)  +
        p1110(p) * exp(- ((g.h0 - h1110(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +  
        p1110(p) * exp(- ((g.h0 - h1101(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1110(p) * exp(- ((g.h0 - h1011(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1110(p) * exp(- ((g.h0 - h0111(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g.h0 - h1100(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g.h0 - h1010(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g.h0 - h0110(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g.h0 - h0011(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g.h0 - h0101(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g.h0 - h1001(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g.h0 - h0001(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g.h0 - h0010(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g.h0 - h0100(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g.h0 - h1000(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2) +
        p0000(p) * exp(- ((g.h0 - h0000(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n))^2))
    return result
end 

function compute_p(p, g::Gain, n::Noise, m1, m2, m3, m4) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p1111(p) * gain_squared(g, h1111(p, g, m1, m2, m3, m4)+ n.σ*x[1]) + 
        p1110(p) * gain_squared(g, h1110(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +  
        p1110(p) * gain_squared(g, h1101(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1110(p) * gain_squared(g, h1011(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1110(p) * gain_squared(g, h0111(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g, h1100(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g, h1010(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g, h0110(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g, h0011(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g, h0101(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g, h1001(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g, h0001(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g, h0010(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g, h0100(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g, h1000(p, g, m1, m2, m3, m4)+ n.σ*x[1]) +
        p0000(p) * gain_squared(g, h0000(p, g, m1, m2, m3, m4)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_p(p, g::Heaviside, n::Noise, m1, m2, m3, m4) 
    result = g.rm^2 * 0.5 * (
        p1111(p) * erfc((g.h0 - h1111(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) + 
        p1110(p) * erfc((g.h0 - h1110(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +  
        p1110(p) * erfc((g.h0 - h1101(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1110(p) * erfc((g.h0 - h1011(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1110(p) * erfc((g.h0 - h0111(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1100(p) * erfc((g.h0 - h1100(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1100(p) * erfc((g.h0 - h1010(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1100(p) * erfc((g.h0 - h0110(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1100(p) * erfc((g.h0 - h0011(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1100(p) * erfc((g.h0 - h0101(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1100(p) * erfc((g.h0 - h1001(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1000(p) * erfc((g.h0 - h0001(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1000(p) * erfc((g.h0 - h0010(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1000(p) * erfc((g.h0 - h0100(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p1000(p) * erfc((g.h0 - h1000(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        p0000(p) * erfc((g.h0 - h0000(p, g, m1, m2 , m3, m4, n) ) / denominator(p, n)))
    return result
end

function push_to_fixed_point!(results, m1, m2, m3, m4, m_val1, m_val2, m_val3, m_val4, bound_up, bound_low, resolution_factor, resolution, r)
    if abs(m1-m_val1)<(bound_up-bound_low)/(resolution_factor*resolution) && 
        abs(m2-m_val2)<(bound_up-bound_low)/(resolution_factor*resolution) && 
        abs(m3-m_val3)<(bound_up-bound_low)/(resolution_factor*resolution) &&
        abs(m4-m_val4)<(bound_up-bound_low)/(resolution_factor*resolution)
        push!(results.M1, m_val1)
        push!(results.M2, m_val2)
        push!(results.M3, m_val3)
        push!(results.M4, m_val4)
        push!(results.R, r)
    end
end

function get_fixed_points(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
    m_val = range( bound_low, stop= bound_up, length = resolution)
    r_val = range( 0, stop= bound_r, length = resolution)
    @show load
    results = Results()
    for m in 1 : resolution
        for k in 1 : resolution
            for j in 1 : resolution
                for i in 1 : resolution
                    if load != 0.
                        for l in 1 : resolution
                            n_temp = Noise(load = load, r = r_val[l], A = p.A)
                            q_func = compute_q(p, g, n_temp, m_val[i], m_val[j], m_val[k], m_val[m])
                            p_func = compute_p(p, g, n_temp, m_val[i], m_val[j], m_val[k], m_val[m]) 
                            D_const = (1 - p.A * q_func)
                            r_fin = p_func /(D_const^2)
                            
                            if abs(r_fin - r_val[l]) < bound_r /(resolution_factor * resolution)
                                if r_val[l] != 0.0            
                                    m1 = nullcline1(p, g, n_temp, m_val[i], m_val[j], m_val[k], m_val[m])
                                    m2 = nullcline2(p, g, n_temp, m_val[i], m_val[j], m_val[k], m_val[m])
                                    m3 = nullcline3(p, g, n_temp, m_val[i], m_val[j], m_val[k], m_val[m])
                                    m4 = nullcline4(p, g, n_temp, m_val[i], m_val[j], m_val[k], m_val[m])
                                    push_to_fixed_point!(results, m1, m2, m3, m4, m_val[i], m_val[j], m_val[k],  m_val[m],  bound_up, bound_low, resolution_factor, resolution, r_val[l])
                                    
                                else 
                                    m1 = nullcline1(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                                    m2 = nullcline2(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                                    m3 = nullcline3(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                                    m4 = nullcline4(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                                    push_to_fixed_point!(results, m1, m2, m3, m4,  m_val[i], m_val[j], m_val[k], m_val[m], bound_up, bound_low, resolution_factor, resolution, 0.)
                                end
                                continue
                            end
                        end
                    else 
                        m1 = nullcline1(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                        m2 = nullcline2(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                        m3 = nullcline3(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                        m4 = nullcline4(p, g, m_val[i], m_val[j], m_val[k], m_val[m])
                        push_to_fixed_point!(results, m1, m2, m3, m4, m_val[i], m_val[j], m_val[k], m_val[m], bound_up, bound_low, resolution_factor, resolution, 0.)
                    end
                end
            end
        end
    end
    return  results.M1, results.M2 , results.M3, results.M4, results.R  
end 

function generate_bifurcation_diagram(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution, resolution_factor = 1, size, load)
    g = Gain( rm = rm, b = b, h0 = h0 )
    c_hat_range = range( 0., stop = h0/rm , length = size )
    println("resolution = ", resolution)
    result = @distributed vcat for i in 1:size
        tmp = DataFrame(C = [], x_p = [], r_p = [], stab = [])
        
        start = time()
        p = NetworkParameters(γ = γ , A = A, C_hat = c_hat_range[i])
        println( "C_hat = ", p.C_hat, "   h0 = ", h0)
        
        M1, M2, M3, M4, R = get_fixed_points(p, g, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
        
        for j in 1: length(M1)
            eigs = real(compute_eigenvalues(p, g, M1[j], M2[j], M3[j], M4[j]))
            if eigs[1] > 0 && eigs[2] > 0 && eigs[3] > 0 && eigs[4] > 0
                push!(tmp, [p.C_hat, M1[j], R[j],  0])
            elseif eigs[1] < 0 && eigs[2] < 0 && eigs[3] < 0 && eigs[4] < 0
                push!(tmp, [p.C_hat, M1[j], R[j],  2])
            else 
                push!(tmp, [p.C_hat, M1[j], R[j],  1])
            end
        end
        elapsed = time() - start
        println("Simulation of point ",  i ," of ", size, " took ", elapsed, " seconds")
        tmp
    end
    CSV.write("Output_4pop/Bifurcation_plot_vs_C_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load).dat", result)
end

end # end of module FourPopulations

module AdaptTwoPop
import ..AttractorNetwork: p1, p11, p10, p00, get_α_β, Heaviside, Gain, dgain_dx,gain_squared, Noise, NoNoise, AdaptNetworkParameters
using PolynomialRoots
using HCubature
using SpecialFunctions
using Roots
using NLsolve
using Distributed, DataFrames, CSV
using Distributions
using Plots
using Waveforms
using LinearAlgebra
using DelimitedFiles

# Define the most used 4 inputs, no noise case
h11(p, g, m1, m2,  mean_phi) = p.A * g.rm * (1. - p.γ) * (m1 + m2) - (p.J0 / p.γ) * mean_phi
h00(p, g, m1, m2,  mean_phi) = - p.γ * p.A * g.rm * (m1 + m2) - (p.J0/p.γ) * mean_phi
h10(p, g, m1, m2,  mean_phi) = p.A * g.rm * (1. - p.γ) * m1 - p.γ * p.A * g.rm * m2 - (p.J0/p.γ) * mean_phi
h01(p, g, m1, m2,  mean_phi) = h10(p, g, m2, m1, mean_phi)

function solve_mean_phi(m1, m2, p, g11::Gain, g10::Gain, g01::Gain, g00::Gain, n::Noise, mean_phi) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * ( 
                    p11(p) * g11(h11(p, g11, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p10(p) * g10(h10(p, g10, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p10(p) * g01(h01(p, g01, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p00(p) * g00(h00(p, g00, m1, m2,  mean_phi) + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function solve_mean_phi(m1, m2, p, g11::Heaviside, g10::Heaviside, g01::Heaviside, g00::Heaviside, n::Noise, mean_phi)
    result = 0.5 * g11.rm * (
        p11(p) * erfc( (g11.h0 - h11(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ)) +
        p10(p) * erfc( (g10.h0 - h10(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ)) +
        p10(p) * erfc( (g01.h0 - h01(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ)) +
        p00(p) * erfc( (g00.h0 - h00(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ)))
    return result
end
function solve_mean_phi(m1, m2, p, g11, g10, g01, g00, n::NoNoise, mean_phi)           
    result = p11(p) * g11(h11(p, g11, m1, m2,  mean_phi)) + 
        p10(p) * (g10(h10(p, g10, m1, m2,  mean_phi)) +  
        g01(h01(p, g01, m1, m2,  mean_phi))) + 
        p00(p) * g00(h00(p, g00, m1, m2,  mean_phi))
    return result 
end
function g!(F, h, m1, m2, p, g11, g10, g01, g00, n)
    ## Here we use Julia's non-linear-solver to find the solution of the mean activity 
    F[1] = solve_mean_phi(m1, m2, p, g11, g10, g01, g00, n, h[1]) - h[1]
end

function compute_mean_phi(p, n, m1, m2, g11, g10, g01, g00)
    r = nlsolve((G, h) -> g!(G, h, m1, m2, p, g11, g10, g01, g00, n), [0.1])
    mean_phi =  r.zero[1]
    return mean_phi
end

# System's variables: m1, m2, p, q
# NOTE: the g11.rm is considered the max firing rates for all four homogeneous populations
function nullcline1(p, g11, g10, g01, g00, n::NoNoise, m1, m2, mean_phi)
    1. / (g11.rm * p.γ * (1. - p.γ)) * ((1. - p.γ) * p11(p) * g11(h11(p, g11, m1, m2,  mean_phi)) + 
        (1. - p.γ) * p10(p) * g10(h10(p, g10, m1, m2,  mean_phi)) -
        p.γ * p10(p) * g01(h01(p, g01, m1, m2,  mean_phi)) -
        p.γ * p00(p) * g00(h00(p, g00, m1, m2,  mean_phi)))
end
function nullcline1(p, g11::Gain, g10::Gain,g01::Gain, g00::Gain, n::Noise, m1, m2,  mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi) * g11.rm * p.γ * (1 - p.γ)) * ( 
        (1 - p.γ) * p11(p) * g11(h11(p, g11, m1, m2,  mean_phi) + n.σ*x[1]) + 
        (1 - p.γ) * p10(p) * g10(h10(p, g10, m1, m2,  mean_phi) + n.σ*x[1]) -
        p.γ * p10(p) * g01(h01(p, g01, m1, m2,  mean_phi) + n.σ*x[1]) -
        p.γ * p00(p) * g00(h00(p, g00, m1, m2,  mean_phi) + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function nullcline1(p, g11::Heaviside, g10::Heaviside, g01::Heaviside, g00::Heaviside, n::Noise, m1, m2, mean_phi)
    result = 0.5 / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ) * p11(p) * erfc( ((g11.h0 - h11(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) +
        (1 - p.γ) * p10(p) * erfc( ((g10.h0 - h10(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) -
        p.γ * p10(p) * erfc( ((g01.h0 - h01(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) -
        p.γ * p00(p) * erfc( ((g00.h0 - h00(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2))
        println("using suspicious function")   #!!!
    return result
end
# NOTE: In order to def. nullcline2 starting from nullcline1, we exchange the m1 and m2, and we also need to exchange g10 and g01, since their internal threshold Theta might be different
nullcline2(p, g11, g10, g01, g00, n,  m1, m2, mean_phi) = nullcline1(p, g11, g01, g10, g00, n,  m2, m1, mean_phi)

compute_phi11(p, g11, noise::NoNoise, m1, m2, mean_phi) = g11(h11(p, g11, m1, m2,  mean_phi))
compute_phi10(p, g10, noise::NoNoise, m1, m2, mean_phi) = g10(h10(p, g10, m1, m2,  mean_phi))
compute_phi01(p, g01, noise::NoNoise, m1, m2, mean_phi) = g01(h01(p, g01, m1, m2,  mean_phi))
compute_phi00(p, g00, noise::NoNoise, m1, m2, mean_phi) = g00(h00(p, g00, m1, m2,  mean_phi))
function compute_phi11(p, g11, n::Noise, m1, m2, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g11(h11(p, g11, m1, m2,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end 
function compute_phi10(p, g10, n::Noise, m1, m2, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g10(h10(p, g10, m1, m2,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end 
function compute_phi01(p, g01, n::Noise, m1, m2, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g01(h01(p, g01, m1, m2,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end 
function compute_phi00(p, g00, n::Noise, m1, m2, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g00(h00(p, g00, m1, m2,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end 

function compute_q(p, g11::Gain, g10::Gain, g01::Gain, g00::Gain, n::Noise, m1, m2, mean_phi) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * ( 
                    p11(p.α, p.β) * dgain_dx(g11, h11(p, g11, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p10(p.β) * dgain_dx(g10, h10(p, g10, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p10(p.β) * dgain_dx(g01, h01(p, g01, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p00(p.α, p.β) * dgain_dx(g00, h00(p, g00, m1, m2,  mean_phi) + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_q(p, g11::Heaviside, g10::Heaviside, g01::Heaviside, g00::Heaviside, n::Noise, m1, m2, mean_phi)
    result = g11.rm/sqrt(2*pi) * (p11(p) * exp(- ((g11.h0 - h11(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2)  +
    p10(p) * exp(-((g10.h0 + h10(p, g10, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) +
    p10(p) * exp(-((g01.h0 + h01(p, g01, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) +
    p00(p) * exp(-((g00.h0 + h00(p, g00, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2))
    return result
end 

function compute_p(p, g11::Gain, g10::Gain, g01::Gain, g00::Gain, n::Noise, m1, m2,  mean_phi) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * ( 
                    p11(p.α, p.β) * gain_squared(g11, h11(p, g11, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p10(p.β) * gain_squared(g10, h10(p, g10, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p10(p.β) * gain_squared(g01, h01(p, g01, m1, m2,  mean_phi) + n.σ*x[1]) +
                    p00(p.α, p.β) * gain_squared(g00, h00(p, g00, m1, m2,  mean_phi) + n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_p(p, g11::Heaviside, g10::Heaviside, g01::Heaviside, g00::Heaviside, n::Noise, m1, m2, mean_phi)
    result = 0.5 * g11.rm^2 * (
        p11(p) * erfc( ((g11.h0 - h11(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) +
        p10(p) * erfc( ((g10.h0 - h10(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) +
        p10(p) * erfc( ((g01.h0 - h01(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2) +
        p00(p) * erfc( ((g00.h0 - h00(p, g11, m1, m2,  mean_phi))/(sqrt(2)*n.σ))^2))
    return result
end

# Here we implement the 4 entries of the Jacobian matrix
function J11(m1, m2, mean_phi,  p, g11, g10, g01, g00, n::NoNoise) 
    - 1 + p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p11(p) * dgain_dx(g11, h11(p, g11, m1, m2, mean_phi)) + 
        (1 - p.γ)^2 * p10(p) * dgain_dx(g10, h10(p, g10, m1, m2, mean_phi)) +
        p.γ^2 * p10(p) * dgain_dx(g01, h01(p, g01, m1, m2, mean_phi)) +
        p.γ^2 * p00(p) * dgain_dx(g00, h00(p, g00, m1, m2, mean_phi)))
end
function J12(m1, m2, mean_phi, p, g11, g10, g01, g00, n::NoNoise) 
    p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p11(p) * dgain_dx(g11, h11(p, g11, m1, m2, mean_phi)) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g10, h10(p, g10, m1, m2, mean_phi)) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g01, h01(p, g01, m1, m2, mean_phi)) +
        p.γ^2 * p00(p) * dgain_dx(g00, h00(p, g00, m1, m2, mean_phi)))
end
function J11(m1, m2, mean_phi, p, g11, g10, g01, g00, n::Noise) 
    integrand = x -> p.A * exp(-x[1]^2 / 2) / (sqrt(2 * pi) * p.γ * (1 - p.γ)) * ( 
        (1 - p.γ)^2 * p11(p) * dgain_dx(g11, h11(p, g11, m1, m2, mean_phi)+ n.σ*x[1]) + 
        (1 - p.γ)^2 * p10(p) * dgain_dx(g10, h10(p, g10, m1, m2, mean_phi)+ n.σ*x[1]) +
        p.γ^2 * p10(p) * dgain_dx(g01, h01(p, g01, m1, m2, mean_phi)+ n.σ*x[1]) +
        p.γ^2 * p00(p) * dgain_dx(g00, h00(p, g00, m1, m2, mean_phi)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = -1 + hcubature(integrand, -bound, bound)[1]
    return result
end
function J12(m1, m2, mean_phi, p, g11, g10, g01, g00, n::Noise) 
    integrand = x -> p.A * exp(-x[1]^2 / 2) / (sqrt(2 * pi) * p.γ * (1 - p.γ)) * ( 
        (1 - p.γ)^2 * p11(p) * dgain_dx(g11, h11(p, g11, m1, m2, mean_phi)+ n.σ*x[1]) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g10, h10(p, g10, m1, m2, mean_phi)+ n.σ*x[1]) -
        p.γ * (1 - p.γ) * p10(p) * dgain_dx(g01, h01(p, g01, m1, m2, mean_phi)+ n.σ*x[1]) +
        p.γ^2 * p00(p) * dgain_dx(g00, h00(p, g00, m1, m2, mean_phi)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
J22(m1, m2, mean_phi, p, g11, g10, g01, g00, n) = J11(m2, m1, mean_phi, p, g11, g01, g10, g00, n)
J21(m1, m2, mean_phi, p, g11, g10, g01, g00, n) = J12(m2, m1, mean_phi, p, g11, g01, g10, g00, n)
function eigenvalues(m1, m2, mean_phi, p, g11, g10, g01, g00, n)
    J = [J11(m1, m2, mean_phi, p, g11, g10, g01, g00, n)  J12(m1, m2, mean_phi, p, g11, g10, g01, g00, n)   ; 
        J21(m1, m2, mean_phi, p, g11, g10, g01, g00, n)  J22(m1, m2, mean_phi, p, g11, g10, g01, g00, n)]
    return eigvals(J)
end 

function update_state(p, g11, g10, g01, g00, m_state, bound_low, bound_up, bound_r, resolution, resolution_factor, load, running_time, h, tau_m)
    r_val = range( 0, stop= bound_r, length = resolution) 
    results = Results()
    m1_old = m_state[1]
    m2_old = m_state[2]
    m1 = 0.
    m2 = 0.
    mean_phi = 0.
    r = 0.
    not_converged = true  # boolean that determines if m1 and m2 have reached convergence (before updating Theta)
    t = 0   # time step
    while not_converged == true
        if load != 0. 
            for k in 1 : resolution
                n_temp = Noise(load=load, r=r_val[k], A = p.A)
                mean_phi = compute_mean_phi(p, n_temp, m1_old, m2_old, g11, g10, g01, g00)
                q_func = compute_q(p, g11, g10, g01, g00, n_temp, m1_old, m2_old , mean_phi)
                p_func = compute_p(p, g11, g10, g01, g00,  n_temp, m1_old, m2_old, mean_phi) 
                D_const = (1 - p.A * q_func)
                r_fin = p_func /(D_const^2)
                
                if abs(r_fin - r_val[k]) < bound_r /(resolution_factor * resolution)
                    if r_val[k] != 0.0            
                        m1 = m1_old + h * (- m1_old + nullcline1(p, g11, g10, g01, g00, n_temp, m1_old, m2_old , mean_phi))/tau_m
                        m2 = m2_old + h * (- m2_old + nullcline2(p, g11, g10, g01, g00, n_temp, m1_old, m2_old , mean_phi))/tau_m
                        r = r_val[k]
                    else 
                        n0 = NoNoise()
                        m1 = m1_old + h * (- m1_old + nullcline1( p, g11, g10, g01, g00,  n0, m1_old, m2_old , mean_phi))/tau_m
                        m2 = m2_old + h * (- m2_old + nullcline2( p, g11, g10, g01, g00,  n0, m1_old, m2_old , mean_phi))/tau_m
                        r = 0.
                    end
                    continue
                end
            end
        else 
            n0 = NoNoise()
            mean_phi = compute_mean_phi(p, n0, m1_old, m2_old, g11, g10, g01, g00)
            m1 = m1_old + h * (- m1_old + nullcline1( p, g11, g10, g01, g00,  n0, m1_old, m2_old , mean_phi))/tau_m
            m2 = m2_old + h * (- m2_old + nullcline2( p, g11, g10, g01, g00,  n0, m1_old, m2_old , mean_phi))/tau_m
            r = 0.
        end
        
        condition = 0.0001
        not_converged = abs( m1_old - m1) > condition * h || abs( m2_old - m2) > condition * h  #abs( m1_old - m1) != 0. || abs( m2_old - m2) != 0. #
        m1_old = copy(m1)  ## !!! to be commented with the while loop
        m2_old = copy(m2)  ## !!! to be commented with the while loop
        t += 1
    end
    return  m1, m2, mean_phi, r
end

function run_dynamics(; rm = 1, A = 1, h0 = 0, γ = 0.1 , b = 1000, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load = 0, C_hat = 0. , final_time = 100, h = 1,  min_J0=0.7, max_J0=1.2, T=0.015, Tth=45, TJ0=25, tau_m = 1)
    Dth = 1.9 * T
    thresh_0 = [h0, h0, h0, h0]
    thresh = [h0, h0, h0, h0]
    m_state = [1, 0]  # we initialize the state of the network in the first pattern
    
    open("Output_2pop_adapt/check_J0.dat","w") do check_J0
        open("Output_2pop_adapt/check_thr.dat","w") do check_thr
            result = @distributed vcat for i in 1:round(Int,final_time/h)
                tmp = DataFrame(time = [], x_p = [], y_p = [],  stab = [])
                running_time = i  * h
                println("running_time = ", running_time)
                
                J0 = (max_J0-min_J0)/2* sin(running_time *(2 * pi)/TJ0 - pi/2) + (max_J0+min_J0)/2                      # Sin waves as in the paper
                #J0 =  (max_J0-min_J0) / 2. * squarewave(2 * pi * running_time/ TJ0 - pi/2.)+ (max_J0 + min_J0) / 2.    # Square wave alternative
                println("J0 = ", J0)
                p = AdaptNetworkParameters(γ = γ , A = A, C_hat = C_hat, J0 = J0)
                
                g11 = Gain( rm = rm, b = b, h0 = thresh[1])
                g10 = Gain( rm = rm, b = b, h0 = thresh[2])  
                g01 = Gain( rm = rm, b = b, h0 = thresh[3])
                g00 = Gain( rm = rm, b = b, h0 = thresh[4])  
                
                X_p, Y_p, mean_phi, r = update_state(p, g11, g10, g01, g00, m_state, bound_low, bound_up, bound_r, resolution, resolution_factor, load, running_time, h, tau_m)
                m_state = [X_p, Y_p]
                noise = NoNoise()
                if load > 0 
                    noise = Noise(load = load, r = r, A = A)
                end
                phi11 = compute_phi11(p, g11, noise, X_p, Y_p, mean_phi)
                phi10 = compute_phi10(p, g10, noise, X_p, Y_p, mean_phi)
                phi01 = compute_phi01(p, g01, noise, X_p, Y_p, mean_phi)
                phi00 = compute_phi00(p, g00, noise, X_p, Y_p, mean_phi)
                phis = [phi11, phi10, phi01, phi00]

                thresh = thresh .- h .* (thresh .- thresh_0 .- (Dth .* phis)) ./ Tth
                write(check_J0, "$(running_time)",  "    ", "$(J0)", "    ", "$(mean_phi)", "    ", "$(phi11)", "   ", "$(phi10)", "    ", "$(phi01 )", "    ", "$(phi00 )","\n ")
                write(check_thr, "$(running_time)",  "    ", "$(thresh[1])",  "    ", "$(thresh[2])","\n ")
        
                ### NOTE TO IMPROVE THE CODE: put this into a separate function!!!
                for j in 1: length(X_p)
                    eigs = real( eigenvalues(X_p[j], Y_p[j], mean_phi, p, g11, g10, g01, g00, noise))  
                    if eigs[1] > 0 && eigs[2] > 0
                        push!(tmp, [running_time, X_p[j], Y_p[j],  0])
                        #println("unstable")
                    elseif (eigs[1] > 0 && eigs[2] < 0) || (eigs[1] < 0 && eigs[2] > 0)
                        push!(tmp, [running_time, X_p[j], Y_p[j],   1])
                        #println("saddle")
                    elseif eigs[1] < 0 && eigs[2] < 0
                        #println("stable")
                        push!(tmp, [running_time, X_p[j], Y_p[j],   2])
                    end
                end

                # The following part Check by printing some PP   - NOTE TO IMPROVE THE CODE: to be separated!!!
                if running_time == h
                    println(" ----------- testing the nullclines -----------")
                    M11, M21= find_nullcline1( p, g11, g10, g01, g00, resolution, resolution_factor, load, bound_r, bound_low, bound_up)
                    open("Output_2pop_adapt/nullcline1_gamma$(γ)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_C_hat$(p.C_hat).dat","w") do null
                        for j in 1:length(M11)
                            write(null, "$(M11[j])   $(M21[j]) \n")
                        end
                    end
                    println(" ------------- end test -------------")
                end
                
                tmp
            end
        CSV.write("Output_2pop_adapt/MF_adapt_vs_time_p2_gamma$(γ)_time_step$(h)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load).dat", result)
        end
    end
end

function find_nullcline1( p, g11, g10, g01, g00, resolution, resolution_factor, load, bound_r, bound_low, bound_up)
    M11 = Float64[]
    M21 = Float64[]
    R = Float64[]
    mean_phi = 0
    MF = Float64[]
    m_val = range( bound_low, stop= bound_up, length = resolution)
    r_val = range( 0, stop= bound_r, length = resolution)

    open("Output_2pop_adapt/quiver.dat","w") do quiver
        for i in 1 : resolution
            m1_temp = Float64[]
            for j in 1 : resolution
                if load != 0.
                    
                    for k in 1 : resolution
                        n_temp = Noise(load=load, r=r_val[k], A = p.A)
                        mean_phi = compute_mean_phi(p, n_temp, m_val[j], m_val[i], g11, g10, g01, g00)
                        q_func = compute_q(p, g11, g10, g01, g00, n_temp, m_val[j], m_val[i], mean_phi)
                        p_func = compute_p(p, g11, g10, g01, g00,  n_temp, m_val[j], m_val[i], mean_phi) 
                        D_const = (1 - p.A * q_func)
                        r_fin = p_func /(D_const^2)
                        
                        if abs(r_fin - r_val[k]) < bound_r /(resolution_factor * resolution)
                            if r_val[k] != 0.0            
                                m1 = nullcline1(p, g11, g10, g01, g00, n_temp, m_val[j], m_val[i], mean_phi)
                                m2 = nullcline2(p, g11, g10, g01, g00, n_temp, m_val[j], m_val[i], mean_phi)
                                push!(m1_temp, m1 - m_val[j])  
                                if i%100 == 0 && j%100 == 0
                                    write(quiver, "$(m_val[j])   $(m_val[i])   $(m1 - m_val[j])  $(m2 - m_val[i])  \n")
                                end
                                if abs(m1-m_val[j])<(bound_up-bound_low)/(resolution_factor*resolution) #length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0    
                                    push!(M11, m_val[j])
                                    push!(M21, m_val[i])
                                    push!(R, r_val[k])
                                    push!(MF, mean_phi)
                                end
                            else 
                                n0 = NoNoise()
                                m1 = nullcline1(p, g11, g10, g01, g00, n0, m_val[j], m_val[i], mean_phi)
                                m2 = nullcline2(p, g11, g10, g01, g00, n0, m_val[j], m_val[i], mean_phi)
                                push!(m1_temp, m1 - m_val[j]) 
                                if i%100 == 0 && j%100 == 0
                                    write(quiver, "$(m_val[j])   $(m_val[i])   $(m1 - m_val[j])  $(m2 - m_val[i])  \n")
                                end
                                if abs(m1-m_val[j])<(bound_up-bound_low)/(resolution_factor*resolution) || length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0    
                                    push!(M11, m_val[j])
                                    push!(M21, m_val[i])
                                    push!(R, 0.)
                                    push!(MF, mean_phi)
                                end
                            end
                            continue
                        end
                    end
                else 
                    n0 = NoNoise()
                    mean_phi = compute_mean_phi(p, n0,  m_val[j], m_val[i], g11, g10, g01, g00)
                    m1 = nullcline1(p, g11, g10, g01, g00, n0, m_val[j], m_val[i], mean_phi)
                    m2 = nullcline2(p, g11, g10, g01, g00, n0, m_val[j], m_val[i], mean_phi)
                    push!(m1_temp, m1 - m_val[j])  
                
                    if i%100 == 0 && j%100 == 0
                        write(quiver, "$(m_val[j])   $(m_val[i])   $(m1 - m_val[j])  $(m2 - m_val[i])  \n")
                    end
                    if  (length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0 )  || (abs(m1-m_val[j])<(bound_up-bound_low)/(resolution_factor*resolution)) 
                        push!(M11, m_val[j])
                        push!(M21, m_val[i])
                        push!(R, 0.)
                        push!(MF, mean_phi)
                    end
                end
            end
        end
    end
    ## NOTE TO IMPROVE THE CODE: the following block of code can be put in a function!!!
    open("Output_2pop_adapt/fp.dat","w") do fp
        x_p, y_p, R_p, MP = get_fixed_points(M11, M21, M21, M11, R, MF, bound_up, bound_low, resolution, resolution_factor)
        for j in 1: length(x_p)
            n = NoNoise()
            if R_p[j] != 0.
                n = Noise(load = load, r = R_p[j], A = A)
            end
            eigs = real( eigenvalues(x_p[j], y_p[j], MP[j], p,  g11, g10, g01, g00, n))
            println("eigenvalues = ", eigs)
            if eigs[1] > 0 && eigs[2] > 0
                write(fp,"$(x_p[j])  $(y_p[j])  0\n")
            elseif (eigs[1] >= 0 && eigs[2] <= 0) || (eigs[1] <= 0 && eigs[2] >= 0)
                write(fp,"$(x_p[j])  $(y_p[j])   1\n")
            elseif eigs[1] < 0 && eigs[2] < 0
                write(fp,"$(x_p[j])  $(y_p[j])   2\n")
            end
        end
    end
    return  M11, M21
end 

### NOTE TO IMPROVE THE CODE: put outside?!!!
struct Results
    M11 :: Vector{Float64} # m1 projection of m1 nullcline
    M21 :: Vector{Float64} # m2 projection of m1 nullcline
    M12 :: Vector{Float64} # m1 projection of m2 nullcline
    M22 :: Vector{Float64} # m2 projection of m2 nullcline
    R1 :: Vector{Float64}  #r coordinate of the first nullcline
    Mean_phi :: Vector{Float64}
end
Results() = Results([], [], [], [], [], [])

function push_to_nullcline!(m1_temp, m1 , m_valj, m_vali, r_valk, mean_phi, results, bound_up, bound_low, resolution_factor, resolution)
    push!(m1_temp, m1 - m_valj)  
    #push!(m2_temp, m2 - m_vali)    
    if length(m1_temp) > 1 &&  m1_temp[end]*m1_temp[end-1] < 0   ||   abs(m1-m_valj)<(bound_up-bound_low)/(resolution_factor*resolution) #
        push!(results.M11, m_valj)
        push!(results.M21, m_vali)
        push!(results.R1, r_valk)
        push!(results.Mean_phi, mean_phi)
    end
end 

### NOTE TO IMPROVE THE CODE: redundant with nullcline 1 !!!
function get_nullclines(p, g11, g10, g01, g00, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
    m_val = range( bound_low, stop= bound_up, length = resolution)
    r_val = range( 0, stop= bound_r, length = resolution)
    results = Results()
    for i in 1 : resolution
        m1_temp = Float64[]
        for j in 1 : resolution
            if load != 0.
                for k in 1 : resolution
                    n_temp = Noise(load=load, r=r_val[k], A = p.A)
                    mean_phi = compute_mean_phi(p, n_temp, m_val[j], m_val[i], g11, g10, g01, g00)
                    q_func = compute_q(p, g11, g10, g01, g00, n_temp, m_val[j], m_val[i], mean_phi)
                    p_func = compute_p(p, g11, g10, g01, g00,  n_temp, m_val[j], m_val[i], mean_phi) 
                    D_const = (1 - p.A * q_func)
                    r_fin = p_func /(D_const^2)
                    if abs(r_fin - r_val[k]) < bound_r /(resolution_factor * resolution)
                        if r_val[k] != 0.0            
                            m1 = nullcline1(p, g11, g10, g01, g00, n_temp, m_val[j], m_val[i], mean_phi)
                            #m2 = nullcline2(p, g, n_temp, h11_val, h10_val, h01_val, h00_val)
                            push_to_nullcline!(m1_temp,  m1 ,  m_val[j], m_val[i], r_fin, mean_phi, results, bound_up, bound_low, resolution_factor, resolution)
                        else 
                            n0 = NoNoise()
                            mean_phi = compute_mean_phi(p, n0,  m_val[j], m_val[i], g11, g10, g01, g00)
                            m1 = nullcline1(p, g11, g10, g01, g00, n0, m_val[j], m_val[i], mean_phi)
                            push_to_nullcline!(m1_temp,  m1 ,  m_val[j], m_val[i], 0, mean_phi, results, bound_up, bound_low, resolution_factor, resolution)
                        end
                        continue
                    end
                end
            else 
                n0 = NoNoise()
                mean_phi = compute_mean_phi(p, n0,  m_val[j], m_val[i], g11, g10, g01, g00)
                m1 = nullcline1(p, g11, g10, g01, g00, n0, m_val[j], m_val[i], mean_phi)
                push_to_nullcline!(m1_temp, m1 ,  m_val[j], m_val[i], 0, mean_phi,  results, bound_up, bound_low, resolution_factor, resolution)
            end
        end
    end
    return  results.M11, results.M21 , results.R1  , results.Mean_phi
end  

### NOTE FOR A MORE POLISHED VERSION OF THE CODE: why are we saving MP into the fp??!!!! if we don't have MP we can have a function in common with TwoPop, but we use MP in the bifurcation just below... maybe there is a better way to implement it
function get_fixed_points(M11, M12, M21, M22, R, mean_phi, bound_up, bound_low, resolution, resolution_factor)
    # the coodinates of the fixed points
    X_p = Float64[]
    Y_p = Float64[]
    R_p = Float64[]
    MP = Float64[]
    for l in 1:length(M11)
        for m in 1:length(M12)
            #if (abs(M12[m]-M11[l])<=(bound_up-bound_low)/(resolution_factor*resolution)) && (abs(M22[m]-M21[l])<=(bound_up-bound_low)/(resolution_factor*resolution)) 
            if (abs(M21[m]-M11[l])<=(bound_up-bound_low)/(resolution_factor*resolution)) && (abs(M11[m]-M21[l])<=(bound_up-bound_low)/(resolution_factor*resolution)) 
                push!(X_p, M11[l])
                push!(Y_p, M22[m])
                push!(R_p, R[m])
                push!(MP, mean_phi[m])
            end
        end
    end
    return X_p, Y_p, R_p, MP
end
### NOTE TO IMPROVE THE CODE: maybe use the same ad TWOPOP?!!! problem with output files
function generate_bifurcation_diagram(; rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution, resolution_factor = 1, size, load, J0)
#function generate_bifurcation_diagram(; rm, A , h0 , γ  , b , bound_low , bound_up , bound_r, resolution, resolution_factor , size, load, J0)
    g11 = Gain( rm = rm, b = b, h0 = h0)
    g10 = Gain( rm = rm, b = b, h0 = h0)  
    g01 = Gain( rm = rm, b = b, h0 = h0)
    g00 = Gain( rm = rm, b = b, h0 = h0) 
    #c_hat_range = range( 0., stop = h0/rm + h0/(10*rm) , length = size )
    c_hat_range = range( 0.5, stop = 0.8 , length = size )
    result = @distributed vcat for i in 1:size
        tmp = DataFrame(C_hat = [], x_p = [], r_p = [], stab = [])
        start = time()
        p = AdaptNetworkParameters(γ = γ , A = A, C_hat = c_hat_range[i], J0 = J0)
        println( "C_hat = ", p.C_hat, "   h0 = ", h0)
    
        M11, M21, R1, mean_phi = get_nullclines(p, g11, g10, g01, g00, bound_low, bound_up, bound_r, resolution, resolution_factor, load)
        X_p, Y_p, R_p, MP = get_fixed_points(M11, M21, M21, M11, R1, mean_phi, bound_up, bound_low, resolution, resolution_factor)
        if size == 5
            open("Output_2pop_adapt/nullcline1_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_C_hat$(round(p.C_hat, digits=2)).dat","w") do null
                for j in 1:length(M11)
                    write(null, "$(M11[j])   $(M21[j]) \n")
                end
                println( "Output_2pop_adapt/nullcline1_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_C_hat$(round(p.C_hat, digits=2)).dat")
            end
            
            ### NOTE FOR A MORE POLISHED VERSION OF THE CODE:  it can be put in a separate function!!!
            open("Output_2pop_adapt/fp_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_C_hat$(round(p.C_hat, digits=2)).dat","w") do fp 
                for j in 1: length(X_p)
                    n = NoNoise()
                    if R_p[j] != 0.
                        n = Noise(load = load, r = R_p[j], A = A)
                    end
                    eigs=real(eigenvalues( X_p[j],Y_p[j], MP[j], p, g11, g10, g01, g00))
                    #eigenvalues(m1, m2, mean_phi, p, g11, g10, g01, g00, n)
                    if eigs[1] > 0 && eigs[2] > 0
                        write(fp, "$(X_p[j])   $(Y_p[j])    $(R_p[j])    0\n")
                    elseif (eigs[1] > 0 && eigs[2] < 0) || (eigs[1] < 0 && eigs[2] > 0)
                        write(fp, "$(X_p[j])   $(Y_p[j])    $(R_p[j])   1\n")
                    elseif eigs[1] < 0 && eigs[2] < 0
                        write(fp, "$(X_p[j])   $(Y_p[j])    $(R_p[j])    2\n")
                    end
                end
            end   
        end
        
        for j in 1: length(X_p)
            n = NoNoise()
            if R_p[j] != 0.
                n = Noise(load = load, r = R_p[j], A = A)
            end
            eigs=real( eigenvalues(X_p[j], Y_p[j], MP[j], p,  g11, g10, g01, g00, n))
            if eigs[1] > 0 && eigs[2] > 0
                push!(tmp, [p.C_hat, X_p[j], R_p[j],  0])
            elseif (eigs[1] >= 0 && eigs[2] <= 0) || (eigs[1] <= 0 && eigs[2] >= 0)
                push!(tmp, [p.C_hat, X_p[j], R_p[j],  1])
            elseif eigs[1] < 0 && eigs[2] < 0
                push!(tmp, [p.C_hat, X_p[j], R_p[j],  2])
            end
        end
        elapsed = time() - start
        println("Simulation of point ",  i ," of ", size, " took ", elapsed, " seconds")
        tmp
    end
    CSV.write("Output_2pop_adapt/Bifurcation_plot_vs_C_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_J0$(J0).dat", result, delim = ' ')
end

function extract_min_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
    (data_cells, header_cells) = readdlm("Output_2pop_adapt/Bifurcation_plot_vs_C_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_J0$(J0).dat", header = true) 
    C_hat = data_cells[:, 1]
    x_p = data_cells[:, 2]
    r_p = data_cells[:, 3]
    stab = data_cells[:, 4]
    C_hat_crit = 0.
    for i in 1:length(x_p)
        if stab[i] == 2.0 && x_p[i] > 0.15 # for the min C_hat
            C_hat_crit = C_hat[i]
            return C_hat_crit
        end
    end
    #return C_hat_crit
end
function extract_max_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
    (data_cells, header_cells) = readdlm("Output_2pop_adapt/Bifurcation_plot_vs_C_gamma$(γ)_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_J0$(J0).dat", header = true) 
    C_hat = data_cells[:, 1]
    x_p = data_cells[:, 2]
    r_p = data_cells[:, 3]
    stab = data_cells[:, 4]
    C_hat_crit = 0.
    for i in 1:length(x_p)
        if stab[i] == 0.0   # for the max C_hat
           C_hat_crit = C_hat[i]
        end
    end
    return C_hat_crit
end

function compute_critical_C_vs_gamma(; rm = 1., A = 1., h0 = 0. , b = 100., bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0, min_J0 = 0.7, max_J0 = 1.2, size_gamma = 10)
    open("Output_2pop_adapt/Cs_vs_gamma_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load)_MinJ0$(min_J0)_MaxJ0$(max_J0).dat","w") do critical
        gamma_vect = range( 0.001, stop = 0.01 , length = size_gamma )
        #min_C_vect = Float64[]
        #max_C_vect = Float64[]
        for i in 1:size_gamma
            γ = gamma_vect[i]
            J0 = max_J0
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_min_crit = extract_min_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(min_C_vect, C_min_crit)
            J0 = min_J0
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_max_crit = extract_max_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(max_C_vect, C_max_crit)
            write(critical, "$(gamma_vect[i])   $(C_min_crit)   $(C_max_crit) \n")
        end 
    end
end
function compute_critical_C_vs_b(; γ = 0.002, rm = 1., A = 1., h0 = 0. ,  bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0, min_J0 = 0.7, max_J0 = 1.2, size_gamma = 50)
    open("Output_2pop_adapt/Cs_vs_b_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_gamma$(γ)_load$(load)_MinJ0$(min_J0)_MaxJ0$(max_J0).dat","w") do critical
        b_vect = range( 5, stop = 500 , length = size_gamma )
        #min_C_vect = Float64[]
        #max_C_vect = Float64[]
        for i in 1:size_gamma
            b = b_vect[i]
            J0 = max_J0
        
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_min_crit = extract_min_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(min_C_vect, C_min_crit)
            J0 = min_J0
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_max_crit = extract_max_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(max_C_vect, C_max_crit)
            write(critical, "$(b_vect[i])   $(C_min_crit)   $(C_max_crit) \n")
        end 
    end
end
function compute_critical_C_vs_h0(; γ = 0.002, rm = 1., A = 1., b = 100. ,  bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0, min_J0 = 0.7, max_J0 = 1.2, size_gamma = 50)
    open("Output_2pop_adapt/Cs_vs_h0_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_gamma$(γ)_b$(b)_load$(load)_MinJ0$(min_J0)_MaxJ0$(max_J0).dat","w") do critical
        h0_vect = range( 0.0, stop = 1. , length = size_gamma )
        #min_C_vect = Float64[]
        #max_C_vect = Float64[]
        for i in 1:size_gamma
            h0 = h0_vect[i]
            J0 = max_J0
        
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_min_crit = extract_min_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(min_C_vect, C_min_crit)
            J0 = min_J0
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_max_crit = extract_max_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(max_C_vect, C_max_crit)
            write(critical, "$(h0_vect[i])   $(C_min_crit)   $(C_max_crit) \n")
        end 
    end
end
function compute_critical_C_vs_J0(; γ = 0.002, rm = 1., A = 1., b = 500., h0 = 0. ,  bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0,  max_J0 = 1.2, size_gamma = 50)
    open("Output_2pop_adapt/Cs_vs_J0_size$(size)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_gamma$(γ)_b$(b)_load$(load)_h0$(h0)_MaxJ0$(max_J0).dat","w") do critical
        min_J0_vect = range( 0.0, stop = 1. , length = size_gamma )
        #min_C_vect = Float64[]
        #max_C_vect = Float64[]
        for i in 1:size
            J0 = max_J0
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_min_crit = extract_min_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(min_C_vect, C_min_crit)
            min_J0 = min_J0_vect[i]
            generate_bifurcation_diagram(rm = rm, A = A , h0 = h0 , γ = γ , b = b , bound_low = bound_low , bound_up = bound_up , bound_r = bound_r , resolution = resolution, resolution_factor = resolution_factor, size = size, load = load, J0 = J0)
            C_max_crit = extract_max_critical_C_hat(rm, A, h0, γ, b, resolution, resolution_factor, size, load, J0)
            #push!(max_C_vect, C_max_crit)
            write(critical, "$(min_J0_vect[i])   $(C_min_crit)   $(C_max_crit) \n")
        end 
    end
end

end  # end of modulde AdaptTwoPop

module AdaptFourPop
import ..AttractorNetwork: p1, get_α_β, Heaviside, Gain, dgain_dx,gain_squared, Noise, NoNoise, AdaptNetworkParameters
using PolynomialRoots
using HCubature
using SpecialFunctions
using Roots
using NLsolve
using Distributed, DataFrames, CSV
using Distributions
using Plots
using Waveforms

### NOTE TO IMPROVE THE CODE:in common with FourPop!!!! 
# Joint probabilities for correlated patterns
p1111(α, β) = α * β^4 + (1 - α) * (1 - β)^4
p1111(p) = p1111(p.α, p.β)
p1110( α, β) = α * β^3 * (1 - β) + (1 - α) * β * (1 - β)^3
p1110(p) = p1110(p.α, p.β)
p1100( α, β) = α * β^2 * (1 - β)^2 + (1 - α) * β^2 * (1 - β)^2
p1100(p) = p1100(p.α, p.β)
p1000( α, β) = α * β * (1 - β)^3 + (1 - α) * β^3 * (1 - β)
p1000(p) = p1000(p.α, p.β)
p0000(α, β) = α * (1 - β)^4 + (1 - α) * β^4
p0000(p) = p0000(p.α, p.β)

# Define 16 inputs, no background noise case
h1111(p, g, m1, m2, m3 , m4, mean_phi) = p.A * g.rm * (1 - p.γ) * (m1 + m2 + m3 + m4)  - (p.J0 / p.γ) * mean_phi
h0000(p, g, m1, m2, m3, m4, mean_phi) = - p.γ * p.A * g.rm * (m1 + m2 + m3 + m4)  - (p.J0 / p.γ) * mean_phi
h1110(p, g, m1, m2, m3, m4, mean_phi) = p.A * g.rm * (1 - p.γ) * (m1 + m2 +m3) - p.γ * p.A * g.rm * m4  - (p.J0 / p.γ) * mean_phi
h1101(p, g, m1, m2, m3, m4, mean_phi) = h1110(p, g, m1, m2, m4, m3,  mean_phi)  
h1011(p, g, m1, m2, m3, m4, mean_phi) = h1110(p, g, m1, m4, m3, m2,  mean_phi)  
h0111(p, g, m1, m2, m3, m4, mean_phi) = h1110(p, g, m4, m2, m3, m1,  mean_phi)  
h1100(p, g, m1, m2, m3, m4, mean_phi) = p.A * g.rm * (1 - p.γ) * (m1 + m2) - p.γ * p.A * g.rm * (m3 + m4)  - (p.J0 / p.γ) * mean_phi
h1010(p, g, m1, m2, m3, m4, mean_phi) = h1100(p, g, m1, m3, m2, m4,  mean_phi)  
h0101(p, g, m1, m2, m3, m4, mean_phi) = h1100(p, g, m4, m2, m3, m1,  mean_phi)  
h1001(p, g, m1, m2, m3, m4, mean_phi) = h1100(p, g, m1, m4, m3, m2,  mean_phi)  
h0110(p, g, m1, m2, m3, m4, mean_phi) = h1100(p, g, m3, m2, m1, m4,  mean_phi) 
h0011(p, g, m1, m2, m3, m4, mean_phi) = h1100(p, g, m3, m4, m1, m2,  mean_phi)  
h1000(p, g, m1, m2, m3, m4, mean_phi) = p.A * g.rm * (1 - p.γ) * (m1) - p.γ * p.A * g.rm * (m2 + m3 + m4)  - (p.J0 / p.γ) * mean_phi
h0100(p, g, m1, m2, m3, m4, mean_phi) = h1000(p, g, m2, m1, m3, m4,  mean_phi)  
h0010(p, g, m1, m2, m3, m4, mean_phi) = h1000(p, g, m3, m2, m1, m4,  mean_phi)  
h0001(p, g, m1, m2, m3, m4, mean_phi) = h1000(p, g, m4, m2, m3, m1,  mean_phi)  

# Here we implement the 10 different entries of the Jacobian matrix  
function J11(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) 
    - 1 + p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p1111(p) * dgain_dx(g1111,h1111(p, g1111, m1, m2, m3, m4,  mean_phi)) + 
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g1110, h1110(p, g1110, m1, m2, m3, m4,  mean_phi)) +  
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g1101, h1101(p, g1101, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g1011, h1011(p, g1011, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1110(p) * dgain_dx(g0111, h0111(p, g0111, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g1100, h1100(p, g1100, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g1010, h1010(p, g1010, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1100(p) * dgain_dx(g0110, h0110(p, g0110, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1100(p) * dgain_dx(g0011, h0011(p, g0011, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1100(p) * dgain_dx(g0101, h0101(p, g0101, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g1001, h1001(p, g1001, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1000(p) * dgain_dx(g0001, h0001(p, g0001, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1000(p) * dgain_dx(g0010, h0010(p, g0010, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1000(p) * dgain_dx(g0100, h0100(p, g0100, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ)^2 * p1000(p) * dgain_dx(g1000, h1000(p, g1000, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p0000(p) * dgain_dx(g0000, h0000(p, g0000, m1, m2, m3, m4,  mean_phi)))
end
function J12(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) 
    p.A / ( p.γ * (1 - p.γ)) * (
        (1 - p.γ)^2 * p1111(p) * dgain_dx(g1111,h1111(p, g1111, m1, m2, m3, m4,  mean_phi)) + 
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g1110, h1110(p, g1110, m1, m2, m3, m4,  mean_phi)) +  
        (1 - p.γ)^2 * p1110(p) * dgain_dx(g1101, h1101(p, g1101, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1110(p) * dgain_dx(g1011, h1011(p, g1011, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1110(p) * dgain_dx(g0111, h0111(p, g0111, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ)^2 * p1100(p) * dgain_dx(g1100, h1100(p, g1100, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g1010, h1010(p, g1010, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g0110, h0110(p, g0110, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1100(p) * dgain_dx(g0011, h0011(p, g0011, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g0101, h0101(p, g0101, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1100(p) * dgain_dx(g1001, h1001(p, g1001, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1000(p) * dgain_dx(g0001, h0001(p, g0001, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p1000(p) * dgain_dx(g0010, h0010(p, g0010, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1000(p) * dgain_dx(g0100, h0100(p, g0100, m1, m2, m3, m4,  mean_phi)) -
        p.γ * (1 - p.γ) * p1000(p) * dgain_dx(g1000, h1000(p, g1000, m1, m2, m3, m4,  mean_phi)) +
        p.γ^2 * p0000(p) * dgain_dx(g0000, h0000(p, g0000, m1, m2, m3, m4,  mean_phi)))
end
J13(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J12(p, g1111, g0000, g1110, g1011, g1101, g0111, g1010, g1100, g0011, g1001, g0110, g0101, g1000, g0010, g0100, g0001, m1, m3, m2, m4,  mean_phi) 
J14(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J12(p, g1111, g0000, g1011, g1101, g1110, g0111, g1001, g1010, g0101, g1100, g0011, g0110, g1000, g0001, g0010, g0100, m1, m4, m3, m2,  mean_phi) 
J22(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J11(p, g1111, g0000, g1110, g1101, g0111, g1011, g1100, g0110, g1001, g0101, g1010, g0011, g0100, g1000, g0010, g0001, m2, m1, m3, m4,  mean_phi)
J33(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J11(p, g1111, g0000, g1110, g0111, g1011, g1101, g0110, g1010, g0101, g0011, g1100, g1001, g0010, g0100, g1000, g0001, m3, m2, m1, m4,  mean_phi)
J44(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J11(p, g1111, g0000, g0111, g1110, g1011, g1101, g0110, g0011, g1100, g1010, g0101, g1001, g0010, g0100, g0001, g1000, m4, m2, m3, m1,  mean_phi)
J23(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J12(p, g1111, g0000, g1110, g1011, g0111, g1101, g1010, g0110, g1001, g0011, g1100, g0101, g0010, g1000, g0100, g0001, m2, m3, m1, m4,  mean_phi)
J24(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J12(p, g1111, g0000, g1011, g1101, g0111, g1110, g1001, g0011, g1100, g0101, g1010, g0110, g0001, g1000, g0010, g0100, m2, m4, m3, m1,  mean_phi)
J34(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi) = J12(p, g1111, g0000, g1011, g0111, g1110, g1101, g0011, g1010, g0101, g0110, g1001, g1100, g0010, g0001, g1000, g0100, m3, m4, m1, m2,  mean_phi)

function compute_eigenvalues(p, g, m1, m2, m3, m4,  mean_phi)
    J = [J11(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)  J12(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)   J13(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)   J14(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi); 
        J12(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)  J22(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)   J23(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)    J24(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi);
        J13(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)  J23(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)   J33(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)    J34(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi);
        J14(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)  J24(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)   J34(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)    J44(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m1, m2, m3, m4,  mean_phi)]
    return eigvals(J)
end 

# System's variables: m1, m2, p, q
function nullcline1(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n::NoNoise, m1, m2, m3, m4,  mean_phi)  
    1 / (g1111.rm * p.γ * (1 - p.γ)) * (
        (1 - p.γ) * p1111(p) * g1111(h1111(p, g1111, m1, m2, m3, m4,  mean_phi)) + 
        (1 - p.γ) * p1110(p) * g1110(h1110(p, g1110, m1, m2, m3, m4,  mean_phi)) +  
        (1 - p.γ) * p1110(p) * g1101(h1101(p, g1101, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1110(p) * g1011(h1011(p, g1011, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1110(p) * g0111(h0111(p, g0111, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1100(p) * g1100(h1100(p, g1100, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1100(p) * g1010(h1010(p, g1010, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1100(p) * g0110(h0110(p, g0110, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1100(p) * g0011(h0011(p, g0011, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1100(p) * g0101(h0101(p, g0101, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1100(p) * g1001(h1001(p, g1001, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1000(p) * g0001(h0001(p, g0001, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1000(p) * g0010(h0010(p, g0010, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1000(p) * g0100(h0100(p, g0100, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1000(p) * g1000(h1000(p, g1000, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p0000(p) * g0000(h0000(p, g0000, m1, m2, m3, m4,  mean_phi)))
end
function nullcline1(p, g1111::Gain, g0000::Gain, g1110::Gain, g1101::Gain, g1011::Gain, g0111::Gain, g1100::Gain, g1010::Gain, g0101::Gain, g1001::Gain, g0110::Gain, g0011::Gain, g1000::Gain, g0100::Gain, g0010::Gain, g0001::Gain, n::Noise, m1, m2, m3, m4,  mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi) * g1111.rm * p.γ * (1 - p.γ)) * (
        (1 - p.γ) * p1111(p) * g1111(h1111(p, g1111, m1, m2, m3, m4,  mean_phi)) + 
        (1 - p.γ) * p1110(p) * g1110(h1110(p, g1110, m1, m2, m3, m4,  mean_phi)) +  
        (1 - p.γ) * p1110(p) * g1101(h1101(p, g1101, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1110(p) * g1011(h1011(p, g1011, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1110(p) * g0111(h0111(p, g0111, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1100(p) * g1100(h1100(p, g1100, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1100(p) * g1010(h1010(p, g1010, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1100(p) * g0110(h0110(p, g0110, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1100(p) * g0011(h0011(p, g0011, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1100(p) * g0101(h0101(p, g0101, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1100(p) * g1001(h1001(p, g1001, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1000(p) * g0001(h0001(p, g0001, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1000(p) * g0010(h0010(p, g0010, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p1000(p) * g0100(h0100(p, g0100, m1, m2, m3, m4,  mean_phi)) +
        (1 - p.γ) * p1000(p) * g1000(h1000(p, g1000, m1, m2, m3, m4,  mean_phi)) -
        p.γ * p0000(p) * g0000(h0000(p, g0000, m1, m2, m3, m4,  mean_phi)))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

denominator(p, n) = sqrt(2)* n.σ
function nullcline1(p, g1111::Heaviside, g0000::Heaviside, g1110::Heaviside, g1101::Heaviside, g1011::Heaviside, g0111::Heaviside, g1100::Heaviside, g1010::Heaviside, g0101::Heaviside, g1001::Heaviside, g0110::Heaviside, g0011::Heaviside, g1000::Heaviside, g0100::Heaviside, g0010::Heaviside, g0001::Heaviside, n::Noise, m1, m2, m3, m4,  mean_phi)
    1 / (p.γ * (1 - p.γ)) * 0.5 * (
        (1 - p.γ) * p1111(p) * erfc((g1111.h0 - h1111(p, g1111, m1, m2 , m3, m4,  mean_phi) ) / denominator(p, n)) + 
        (1 - p.γ) * p1110(p) * erfc((g1110.h0 - h1110(p, g1110, m1, m2 , m3, m4,  mean_phi) ) / denominator(p, n)) +  
        (1 - p.γ) * p1110(p) * erfc((g1101.h0 - h1101(p, g1101, m1, m2 , m3, m4,  mean_phi) ) / denominator(p, n)) +
        (1 - p.γ) * p1110(p) * erfc((g1011.h0 - h1011(p, g1011, m1, m2 , m3, m4,  mean_phi) ) / denominator(p, n)) -
        p.γ * p1110(p) * erfc((g0111.h0 - h0111(p, g0111, m1, m2 , m3, m4, n) ) / denominator(p, n)) +
        (1 - p.γ) * p1100(p) * erfc((g1100.h0 - h1100(p, g1100, m1, m2 , m3, m4,  mean_phi) ) / denominator(p, n)) +
        (1 - p.γ) * p1100(p) * erfc((g1010.h0 - h1010(p, g1010, m1, m2 , m3, m4,  mean_phi)  )/ denominator(p, n)) -
        p.γ * p1100(p) * erfc((g0110.h0 - h0110(p, g0110, m1, m2 , m3, m4,  mean_phi) ) / denominator(p, n)) -
        p.γ * p1100(p) * erfc((g0011.h0 - h0011(p, g0011, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) -
        p.γ * p1100(p) * erfc((g0101.h0 - h0101(p, g0101, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        (1 - p.γ) * p1100(p) * erfc((g1001.h0 - h1001(p, g1001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) -
        p.γ * p1000(p) * erfc((g0001.h0 - h0001(p, g0001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) -
        p.γ * p1000(p) * erfc((g0010.h0 - h0010(p, g0010, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) -
        p.γ * p1000(p) * erfc((g0100.h0 - h0100(p, g0100, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        (1 - p.γ) * p1000(p) * erfc((g1000.h0 - h1000(p, g1000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) -
        p.γ * p0000(p) * erfc((g0000.h0 - h0000(p, g0000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)))
    return result
end
nullcline2(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n,  m1, m2, m3, m4, mean_phi) = nullcline1(p, g1111, g0000, g1110, g1101, g0111, g1011, g1100, g0110, g1001, g0101, g1010, g0011, g0100, g1000, g0010, g0001, n,  m2, m1, m3, m4, mean_phi) #exchange m1 with m2
nullcline3(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n,  m1, m2, m3, m4, mean_phi) = nullcline1(p, g1111, g0000, g1110, g0111, g1011, g1101, g0110, g1010, g0101, g0011, g1100, g1001, g0010, g0100, g1000, g0001, n,  m3, m2, m1, m4, mean_phi)
nullcline4(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n,  m1, m2, m3, m4, mean_phi) = nullcline1(p, g1111, g0000, g0111, g1101, g1011, g1110, g0101, g0011, g1100, g1001, g0110, g1010, g0001, g0100, g0010, g1000, n,  m4, m2, m3, m1, mean_phi)

function compute_q(p, g1111::Gain, g0000::Gain, g1110::Gain, g1101::Gain, g1011::Gain, g0111::Gain, g1100::Gain, g1010::Gain, g0101::Gain, g1001::Gain, g0110::Gain, g0011::Gain, g1000::Gain, g0100::Gain, g0010::Gain, g0001::Gain, n::Noise, m1, m2, m3, m4, mean_phi) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p1111(p) * dgain_dx(g1111, h1111(p, g1111, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) + 
        p1110(p) * dgain_dx(g1110, h1110(p, g1110, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +  
        p1110(p) * dgain_dx(g1101, h1101(p, g1101, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1110(p) * dgain_dx(g1011, h1011(p, g1011, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1110(p) * dgain_dx(g0111, h0111(p, g0111, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g1100, h1100(p, g1100, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g1010, h1010(p, g1010, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g0110, h0110(p, g0110, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g0011, h0011(p, g0011, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g0101, h0101(p, g0101, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * dgain_dx(g1001, h1001(p, g1001, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g0001, h0001(p, g0001, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g0010, h0010(p, g0010, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g0100, h0100(p, g0100, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * dgain_dx(g1000, h1000(p, g1000, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p0000(p) * dgain_dx(g0000, h0000(p, g, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_q(p, g1111::Heaviside, g0000::Heaviside, g1110::Heaviside, g1101::Heaviside, g1011::Heaviside, g0111::Heaviside, g1100::Heaviside, g1010::Heaviside, g0101::Heaviside, g1001::Heaviside, g0110::Heaviside, g0011::Heaviside, g1000::Heaviside, g0100::Heaviside, g0010::Heaviside, g0001::Heaviside, n::Noise, m1, m2, m3, m4, mean_phi)
    result = g1111.rm/sqrt(2*pi) * (
        p1111(p) * exp(- ((g1111.h0 - h1111(p, g1111, m1, m2, m3, m4, mean_phi)) / denominator(p, n))^2)  +
        p1110(p) * exp(- ((g1110.h0 - h1110(p, g1110, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +  
        p1110(p) * exp(- ((g1101.h0 - h1101(p, g1101, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1110(p) * exp(- ((g1011.h0 - h1011(p, g1011, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1110(p) * exp(- ((g0111.h0 - h0111(p, g0111, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g1100.h0 - h1100(p, g1100, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g1010.h0 - h1010(p, g1010, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g0110.h0 - h0110(p, g0110, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g0011.h0 - h0011(p, g0011, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g0101.h0 - h0101(p, g0101, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1100(p) * exp(- ((g1001.h0 - h1001(p, g1001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g0001.h0 - h0001(p, g0001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g0010.h0 - h0010(p, g0010, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g0100.h0 - h0100(p, g0100, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p1000(p) * exp(- ((g1000.h0 - h1000(p, g1000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2) +
        p0000(p) * exp(- ((g0000.h0 - h0000(p, g0000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n))^2))
    return result
end 

function compute_p(p, g1111::Gain, g0000::Gain, g1110::Gain, g1101::Gain, g1011::Gain, g0111::Gain, g1100::Gain, g1010::Gain, g0101::Gain, g1001::Gain, g0110::Gain, g0011::Gain, g1000::Gain, g0100::Gain, g0010::Gain, g0001::Gain, n::Noise, m1, m2, m3, m4, mean_phi) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p1111(p) * gain_squared(g1111, h1111(p, g1111, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) + 
        p1110(p) * gain_squared(g1110, h1110(p, g1110, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +  
        p1110(p) * gain_squared(g1101, h1101(p, g1101, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1110(p) * gain_squared(g1011, h1011(p, g1011, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1110(p) * gain_squared(g0111, h0111(p, g0111, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g1100, h1100(p, g1100, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g1010, h1010(p, g1010, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g0110, h0110(p, g0110, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g0011, h0011(p, g0011, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g0101, h0101(p, g0101, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * gain_squared(g1001, h1001(p, g1001, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g0001, h0001(p, g0001, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g0010, h0010(p, g0010, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g0100, h0100(p, g0100, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * gain_squared(g1000, h1000(p, g1000, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p0000(p) * gain_squared(g0000, h0000(p, g0000, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function compute_p(p, g1111::Heaviside, g0000::Heaviside, g1110::Heaviside, g1101::Heaviside, g1011::Heaviside, g0111::Heaviside, g1100::Heaviside, g1010::Heaviside, g0101::Heaviside, g1001::Heaviside, g0110::Heaviside, g0011::Heaviside, g1000::Heaviside, g0100::Heaviside, g0010::Heaviside, g0001::Heaviside, n::Noise, m1, m2, m3, m4, mean_phi) 
    g1111.rm^2 * 0.5 * (
        p1111(p) * erfc((g1111.h0 - h1111(p, g1111, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) + 
        p1110(p) * erfc((g1110.h0 - h1110(p, g1110, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +  
        p1110(p) * erfc((g1101.h0 - h1101(p, g1101, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1110(p) * erfc((g1011.h0 - h1011(p, g1011, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1110(p) * erfc((g0111.h0 - h0111(p, g0111, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g1100.h0 - h1100(p, g1100, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g1010.h0 - h1010(p, g1010, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g0110.h0 - h0110(p, g0110, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g0011.h0 - h0011(p, g0011, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g0101.h0 - h0101(p, g0101, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g1001.h0 - h1001(p, g1001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g0001.h0 - h0001(p, g0001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g0010.h0 - h0010(p, g0010, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g0100.h0 - h0100(p, g0100, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g1000.h0 - h1000(p, g1000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p0000(p) * erfc((g0000.h0 - h0000(p, g0000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)))
    return result
end

function solve_mean_phi(m1, m2, m3, m4, p, g1111::Gain, g0000::Gain, g1110::Gain, g1101::Gain, g1011::Gain, g0111::Gain, g1100::Gain, g1010::Gain, g0101::Gain, g1001::Gain, g0110::Gain, g0011::Gain, g1000::Gain, g0100::Gain, g0010::Gain, g0001::Gain, n::Noise, mean_phi) 
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * (
        p1111(p) * g1111(h1111(p, g1111, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) + 
        p1110(p) * g1110(h1110(p, g1110, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +  
        p1110(p) * g1101(h1101(p, g1101, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1110(p) * g1011(h1011(p, g1011, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1110(p) * g0111(h0111(p, g0111, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * g1100(h1100(p, g1100, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * g1010(h1010(p, g1010, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * g0110(h0110(p, g0110, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * g0011(h0011(p, g0011, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * g0101(h0101(p, g0101, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1100(p) * g1001(h1001(p, g1001, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * g0001(h0001(p, g0001, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * g0010(h0010(p, g0010, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * g0100(h0100(p, g0100, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p1000(p) * g1000(h1000(p, g1000, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]) +
        p0000(p) * g0000(h0000(p, g0000, m1, m2, m3, m4, mean_phi)+ n.σ*x[1]))
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
function solve_mean_phi(m1, m2, m3, m4, p, g1111::Heaviside, g0000::Heaviside, g1110::Heaviside, g1101::Heaviside, g1011::Heaviside, g0111::Heaviside, g1100::Heaviside, g1010::Heaviside, g0101::Heaviside, g1001::Heaviside, g0110::Heaviside, g0011::Heaviside, g1000::Heaviside, g0100::Heaviside, g0010::Heaviside, g0001::Heaviside, n::Noise, mean_phi)
    result = 0.5 * g1111.rm * (
        p1111(p) * erfc((g1111.h0 - h1111(p, g1111, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) + 
        p1110(p) * erfc((g1110.h0 - h1110(p, g1110, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +  
        p1110(p) * erfc((g1101.h0 - h1101(p, g1101, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1110(p) * erfc((g1011.h0 - h1011(p, g1011, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1110(p) * erfc((g0111.h0 - h0111(p, g0111, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g1100.h0 - h1100(p, g1100, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g1010.h0 - h1010(p, g1010, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g0110.h0 - h0110(p, g0110, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g0011.h0 - h0011(p, g0011, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g0101.h0 - h0101(p, g0101, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1100(p) * erfc((g1001.h0 - h1001(p, g1001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g0001.h0 - h0001(p, g0001, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g0010.h0 - h0010(p, g0010, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g0100.h0 - h0100(p, g0100, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p1000(p) * erfc((g1000.h0 - h1000(p, g1000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)) +
        p0000(p) * erfc((g0000.h0 - h0000(p, g0000, m1, m2 , m3, m4, mean_phi) ) / denominator(p, n)))
    return result
end

function solve_mean_phi(m1, m2, m3, m4, p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n::NoNoise, mean_phi)       
    result = p1111(p) * g1111(h1111(p, g1111, m1, m2, m3, m4,  mean_phi)) + 
        p1110(p) * g1110(h1110(p, g1110, m1, m2, m3, m4,  mean_phi)) +  
        p1110(p) * g1101(h1101(p, g1101, m1, m2, m3, m4,  mean_phi)) +
        p1110(p) * g1011(h1011(p, g1011, m1, m2, m3, m4,  mean_phi)) +
        p1110(p) * g0111(h0111(p, g0111, m1, m2, m3, m4,  mean_phi)) +
        p1100(p) * g1100(h1100(p, g1100, m1, m2, m3, m4,  mean_phi)) +
        p1100(p) * g1010(h1010(p, g1010, m1, m2, m3, m4,  mean_phi)) +
        p1100(p) * g0110(h0110(p, g0110, m1, m2, m3, m4,  mean_phi)) +
        p1100(p) * g0011(h0011(p, g0011, m1, m2, m3, m4,  mean_phi)) +
        p1100(p) * g0101(h0101(p, g0101, m1, m2, m3, m4,  mean_phi)) +
        p1100(p) * g1001(h1001(p, g1001, m1, m2, m3, m4,  mean_phi)) +
        p1000(p) * g0001(h0001(p, g0001, m1, m2, m3, m4,  mean_phi)) +
        p1000(p) * g0010(h0010(p, g0010, m1, m2, m3, m4,  mean_phi)) +
        p1000(p) * g0100(h0100(p, g0100, m1, m2, m3, m4,  mean_phi)) +
        p1000(p) * g1000(h1000(p, g1000, m1, m2, m3, m4,  mean_phi)) +
        p0000(p) * g0000(h0000(p, g0000, m1, m2, m3, m4,  mean_phi))
    return result 
end
function g!(F, h, m1, m2, m3, m4, p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n)
    ## Here we use Julia's non-linear-solver to find the solution of the mean activity 
    #println("no noise case")
    F[1] = solve_mean_phi(m1, m2, m3, m4, p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n, h[1]) - h[1]
end

function compute_mean_phi(p, n, m1, m2, m3, m4, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001)
    r = nlsolve((G, h) -> g!(G, h, m1, m2, m3, m4, p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n), [0.1])
    mean_phi =  r.zero[1]
    return mean_phi
end

compute_phi1111(p, g1111, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1111(h1111(p, g1111, m1, m2, m3, m4,  mean_phi))
function compute_phi1111(p, g1111, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1111(h1111(p, g1111, m1, m2, m3, m4,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0000(p, g0000, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0000(h0000(p, g0000, m1, m2, m3, m4, mean_phi)) 
function compute_phi0000(p, g0000, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0000(h0000(p, g0000, m1, m2, m3, m4,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end 

compute_phi1110(p, g1110, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1110(h1110(p, g1110, m1, m2, m3, m4, mean_phi))
function compute_phi1110(p, g1110, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1110(h1110(p, g1110, m1, m2, m3, m4,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end 

compute_phi1101(p, g1101, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1101(h1101(p, g1101, m1, m2, m3, m4, mean_phi))
function compute_phi1101(p, g1101, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1101(h1101(p, g1101, m1, m2, m3, m4,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi1011(p, g1011, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1011(h1011(p, g1011, m1, m2, m3, m4, mean_phi))
function compute_phi1011(p, g1011, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1011(h1011(p, g1011, m1, m2, m3, m4,  mean_phi)+ n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0111(p, g0111, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0111(h0111(p, g0111, m1, m2, m3, m4, mean_phi))
function compute_phi0111(p, g0111, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0111(h0111(p, g0111, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi1100(p, g1100, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1100(h1100(p, g1100, m1, m2, m3, m4, mean_phi))
function compute_phi1100(p, g1100, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1100(h1100(p, g1100, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi1010(p, g1010, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1010(h1010(p, g1010, m1, m2, m3, m4, mean_phi))
function compute_phi1010(p, g1010, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1010(h1010(p, g1010, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0101(p, g0101, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0101(h0101(p, g0101, m1, m2, m3, m4, mean_phi))
function compute_phi0101(p, g0101, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0101(h0101(p, g0101, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi1001(p, g1001, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1001(h1001(p, g1001, m1, m2, m3, m4, mean_phi))
function compute_phi1001(p, g1001, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1001(h1001(p, g1001, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0110(p, g0110, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0110(h0110(p, g0110, m1, m2, m3, m4, mean_phi))
function compute_phi0110(p, g0110, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0110(h0110(p, g0110, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0011(p, g0011, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0011(h0011(p, g0011, m1, m2, m3, m4, mean_phi))
function compute_phi0011(p, g0011, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0011(h0011(p, g0011, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi1000(p, g1000, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g1000(h1000(p, g1000, m1, m2, m3, m4, mean_phi))
function compute_phi1000(p, g1000, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g1000(h1000(p, g1000, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0100(p, g0100, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0100(h0100(p, g0100, m1, m2, m3, m4, mean_phi))
function compute_phi0100(p, g0100, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0100(h0100(p, g0100, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0010(p, g0010, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0010(h0010(p, g0010, m1, m2, m3, m4, mean_phi))
function compute_phi0010(p, g0010, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0010(h0010(p, g0010, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end

compute_phi0001(p, g0001, noise::NoNoise, m1, m2, m3, m4, mean_phi) = g0001(h0001(p, g0001, m1, m2, m3, m4, mean_phi))
function compute_phi0001(p, g0001, noise::Noise, m1, m2, m3, m4, mean_phi)
    integrand = x -> exp(-x[1]^2 / 2) / (sqrt(2 * pi)) * g0001(h0001(p, g0001, m1, m2, m3, m4,  mean_phi) + n.σ*x[1])
    bound = [max(5 * n.σ, 3)]
    result = hcubature(integrand, -bound, bound)[1]
    return result
end
# why is this deleted?? !!!!
"""
struct Results
    M1 :: Vector{Float64}  
    M2 :: Vector{Float64}  
    M3 :: Vector{Float64}  
    M4 :: Vector{Float64}  
    R :: Vector{Float64}  
end
Results() = Results([], [], [], [], [])
"""
function update_state(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m_state, bound_low, bound_up, bound_r, resolution, resolution_factor, load, running_time, h, tau_m)
    r_val = range( 0, stop= bound_r, length = resolution) 
    #results = Results()
    m1_old = m_state[1]
    m2_old = m_state[2]
    m3_old = m_state[3]
    m4_old = m_state[4]
    m1 = 0.
    m2 = 0.
    m3 = 0.
    m4 = 0.
    mean_phi = 0.
    r = 0.
    a = true
    j = 0
    while a == true
        if load != 0. 
            for k in 1 : resolution
                n_temp = Noise(load=load, r=r_val[k], A = p.A)
                mean_phi = compute_mean_phi(p, n_temp, m1_old, m2_old, m3_old, m4_old, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001)
                q_func = compute_q(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n_temp, m1_old, m2_old, m3_old, m4_old, mean_phi)
                p_func = compute_p(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n_temp, m1_old, m2_old, m3_old, m4_old, mean_phi) 
                D_const = (1 - p.A * q_func)
                r_fin = p_func /(D_const^2)
                    
                if abs(r_fin - r_val[k]) < bound_r /(resolution_factor * resolution)
                    if r_val[k] != 0.0            
                        m1 = m1_old + h * (- m1_old + nullcline1(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n_temp, m1_old, m2_old, m3_old, m4_old, mean_phi))/tau_m
                        m2 = m2_old + h * (- m2_old + nullcline2(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n_temp, m1_old, m2_old, m3_old, m4_old, mean_phi))/tau_m
                        m3 = m3_old + h * (- m3_old + nullcline3(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n_temp, m1_old, m2_old, m3_old, m4_old, mean_phi))/tau_m
                        m4 = m4_old + h * (- m4_old + nullcline4(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, n_temp, m1_old, m2_old, m3_old, m4_old, mean_phi))/tau_m
                        r = r_val[k]
                    else 
                        n0 = NoNoise()
                        m1 = m1_old + h * (- m1_old + nullcline1( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old , mean_phi))/tau_m
                        m2 = m2_old + h * (- m2_old + nullcline2( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old , mean_phi))/tau_m
                        m3 = m3_old + h * (- m3_old + nullcline3( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old , mean_phi))/tau_m
                        m4 = m4_old + h * (- m4_old + nullcline4( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old , mean_phi))/tau_m
                        r = 0.
                    end
                    continue
                end
            end
        else 
            n0 = NoNoise()
            mean_phi = compute_mean_phi(p, n0, m1_old, m2_old, m3_old, m4_old, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001)
            m1 = m1_old + h * (- m1_old + nullcline1( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old  , mean_phi))/tau_m
            m2 = m2_old + h * (- m2_old + nullcline2( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old  , mean_phi))/tau_m
            m3 = m3_old + h * (- m3_old + nullcline3( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old  , mean_phi))/tau_m
            m4 = m4_old + h * (- m4_old + nullcline4( p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001,  n0, m1_old, m2_old, m3_old, m4_old  , mean_phi))/tau_m
            r = 0.
        end
        condition = 0.0001
        a = abs( m1_old - m1) > condition * h || abs( m2_old - m2) > condition * h || abs( m3_old - m3) > condition * h || abs( m4_old - m4) > condition * h  #abs( m1_old - m1) != 0. || abs( m2_old - m2) != 0. #
        m1_old = copy(m1)  ## !!! to be commented with the while loop
        m2_old = copy(m2)  ## !!! to be commented with the while loop
        m3_old = copy(m3)  ## !!! to be commented with the while loop
        m4_old = copy(m4)  ## !!! to be commented with the while loop
        j += 1
    end
    return  m1, m2, m3, m4, mean_phi, r
end

function run_dynamics(; rm = 1, A = 1, h0 = 0, γ = 0.1 , b = 1000, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load = 0, C_hat = 0. , final_time = 100, h = 1,  min_J0=0.7, max_J0=1.2, T=0.015, Tth=45, TJ0=25, tau_m = 1)
    Dth = 1.9 * T
    thresh_0 = [h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0]
    thresh = [h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0, h0]

    m_state = [1, 0, 0, 0]  # we initialize the state of the network in the first pattern
    # The next file is used to add J0 in the plots (### NOTE TO IMPROVE THE CODE: maybe add it in the normal output file!!!)
    open("Output_4pop_adapt/J0_vs_time.dat","w") do check_J0   
        # The next file can be useful to check the time evolution of thresholds
        open("Output_4pop_adapt/thresholds_vs_time.dat","w") do check_thr  
            result = @distributed vcat for i in 1:round(Int,final_time/h)
                tmp = DataFrame(time = [], m1 = [], m2 = [], m3 = [], m4 = [], stab = [])
                running_time = i  * h
                
                J0 = (max_J0-min_J0)/2* sin(running_time *(2 * pi)/TJ0 - pi/2) + (max_J0+min_J0)/2                      # Sinusoidal periodic inhibition
                #J0 =  (max_J0-min_J0) / 2. * squarewave(2 * pi * running_time/ TJ0 - pi/2.)+ (max_J0 + min_J0) / 2.    # Squared wave periodic inhibition
        
                p = AdaptNetworkParameters(γ = γ , A = A, C_hat = C_hat, J0 = J0)
                g1111 = Gain( rm = rm, b = b, h0 = thresh[1])
                g0000 = Gain( rm = rm, b = b, h0 = thresh[2])
                g1110 = Gain( rm = rm, b = b, h0 = thresh[3])
                g1101 = Gain( rm = rm, b = b, h0 = thresh[4])
                g1011 = Gain( rm = rm, b = b, h0 = thresh[5])
                g0111 = Gain( rm = rm, b = b, h0 = thresh[6])
                g1100 = Gain( rm = rm, b = b, h0 = thresh[7])
                g1010 = Gain( rm = rm, b = b, h0 = thresh[8])
                g0101 = Gain( rm = rm, b = b, h0 = thresh[9])
                g1001 = Gain( rm = rm, b = b, h0 = thresh[10])
                g0110 = Gain( rm = rm, b = b, h0 = thresh[11])
                g0011 = Gain( rm = rm, b = b, h0 = thresh[12])
                g1000 = Gain( rm = rm, b = b, h0 = thresh[13])
                g0100 = Gain( rm = rm, b = b, h0 = thresh[14])
                g0010 = Gain( rm = rm, b = b, h0 = thresh[15])
                g0001 = Gain( rm = rm, b = b, h0 = thresh[16])
                gains = [g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001]
                
                ### NOTE TO IMPROVE THE CODE: why not to pass the gain vector define above?!!!
                m1, m2, m3, m4, mean_phi, r = update_state(p, g1111, g0000, g1110, g1101, g1011, g0111, g1100, g1010, g0101, g1001, g0110, g0011, g1000, g0100, g0010, g0001, m_state, bound_low, bound_up, bound_r, resolution, resolution_factor, load, running_time, h, tau_m)
                
                m_state = [m1, m2, m3, m4]
                noise = NoNoise()
                if r > 0 
                    noise = Noise(load = load, r = r, A = A)
                end
                
                phi1111 = compute_phi1111(p, g1111, noise, m1, m2, m3, m4, mean_phi)
                phi0000 = compute_phi0000(p, g0000, noise, m1, m2, m3, m4, mean_phi)
                phi1110 = compute_phi1110(p, g1110, noise, m1, m2, m3, m4, mean_phi)
                phi1101 = compute_phi1101(p, g1101, noise, m1, m2, m3, m4, mean_phi)
                phi1011 = compute_phi1011(p, g1011, noise, m1, m2, m3, m4, mean_phi)
                phi0111 = compute_phi0111(p, g0111, noise, m1, m2, m3, m4, mean_phi)
                phi1100 = compute_phi1100(p, g1100, noise, m1, m2, m3, m4, mean_phi)
                phi1010 = compute_phi1010(p, g1010, noise, m1, m2, m3, m4, mean_phi)
                phi0101 = compute_phi0101(p, g0101, noise, m1, m2, m3, m4, mean_phi)
                phi1001 = compute_phi1001(p, g1001, noise, m1, m2, m3, m4, mean_phi)
                phi0110 = compute_phi0110(p, g0110, noise, m1, m2, m3, m4, mean_phi)
                phi0011 = compute_phi0011(p, g0011, noise, m1, m2, m3, m4, mean_phi)
                phi1000 = compute_phi1000(p, g1000, noise, m1, m2, m3, m4, mean_phi)
                phi0100 = compute_phi0100(p, g0100, noise, m1, m2, m3, m4, mean_phi)
                phi0010 = compute_phi0010(p, g0010, noise, m1, m2, m3, m4, mean_phi)
                phi0001 = compute_phi0001(p, g0001, noise, m1, m2, m3, m4, mean_phi)
                phis = [phi1111, phi0000, phi1110, phi1101, phi1011, phi0111, phi1100, phi1010, phi0101, phi1001, phi0110, phi0011, phi1000, phi0100, phi0010, phi0001]
                recomputed_mean_phi = p1111(p) * phi1111 + p0000(p) * phi0000 + p1110(p) * phi1110 +
                    p1110(p) * phi1101 + p1110(p) * phi1011 + p1110(p) * phi0111 + p1100(p) * phi1100 + p1100(p) * phi1010 +
                    p1100(p) * phi0101 + p1100(p) * phi1001 + p1100(p) * phi0110 + p1100(p) * phi0011 + p1000(p) * phi1000 +
                    p1000(p) * phi0100 + p1000(p) * phi0010 + p1000(p) * phi0001

                thresh = thresh .- h .* (thresh .- thresh_0 .- (Dth .* phis)) ./ Tth
                write(check_J0, "$(running_time)",  "    ", "$(J0)", "    ", "$(mean_phi)","    "," $(recomputed_mean_phi)", "    ", "$(phi1111)", "   ", "$(phi0000)", "    ","$(phi1110)", "    ","$( phi1101)", "    ","$(phi1011)", "    ","$(phi0111)", "    ","$(phi1100)", "    ","$(phi1010)", "    ","$(phi0101)", "    ","$(phi1001)", "    ","$(phi0110)", "    ","$(phi0011)", "    ","$(phi1000)", "    ","$(phi0100)", "    ","$(phi0010)", "    ","$(phi0001)","\n ")
                write(check_thr, "$(running_time)",  "    ", "$(thresh[1])",  "    ", "$(thresh[2])","\n ")
                
                for j in 1: length(m1)
                    push!(tmp, [running_time, m1[j], m2[j], m3[j], m4[j],  0])
                    """
                    eigs = real( eigenvalues(m1[j], m2[j], m3[j], m4[j], p, g11, g10, g01, g00, mean_phi))  
                    if eigs[1] > 0 && eigs[2] > 0 && eigs[3] > 0 && eigs[4] > 0
                        push!(tmp, [running_time, m1[j], m2[j], m3[j], m4[j],  0])
                        #println("unstable")
                    elseif eigs[1] < 0 && eigs[2] < 0 && eigs[3] < 0 && eigs[4] < 0
                        push!(tmp, [running_time, m1[j], m2[j], m3[j], m4[j],   2])
                        #println("stable")
                    else 
                        #println("saddle")
                        push!(tmp, [running_time, m1[j], m2[j], m3[j], m4[j],   1])
                    end
                    """
                end
                tmp
            end
            CSV.write("Output_4pop_adapt/MF_adapt_vs_time_p4_gamma$(γ)_time_step$(h)_resolution$(resolution)_factor$(resolution_factor)_rm$(rm)_A$(A)_h0$(h0)_b$(b)_load$(load).dat", result)
        end
    end
end

end  # end of modulde AdaptFourPop

module FullSim
import ..AttractorNetwork: get_α_β , Gain, Heaviside, generate_binary_patterns, generate_binary_patterns_fix_sparseness, weight_matrix, diluted_weight_matrix, overlap, generate_random_binary_patterns_2contex_3people
using StatsBase
using LinearAlgebra
using Statistics
using Distributions
using Random

struct NetworkParameters
    N::Int64
    P::Int64
    n::Int64  ## number of correlated signals
    γ::Float64
    C_hat::Float64
    A::Float64
    α::Float64
    β::Float64
    tau::Float64
end
function NetworkParameters(; N, P, n, γ = 1/1000, A = 1, C_hat, tau = 1)   
    α, β = get_α_β(γ, C_hat)
    NetworkParameters(N, P, n, γ, C_hat, A, α, β, tau)
end
function  input_field(p, g, p_mat, dilution, W, S, current_overlap, ext_input)  
    #println("(p_mat .- p.γ) * current_overlap = ", size((p_mat .- p.γ) * current_overlap))
    #return p.A  * g.rm  * (p_mat .- p.γ) * current_overlap .+ ext_input
    return W * S .+ ext_input
end 

function  update_state_Euler(p, g, h, p_mat,  dilution, W,  S, current_overlap, ext_input)
    S_old = copy(S)
    S_new = S_old .+ h.*(- S_old .+ g(input_field( p, g, p_mat, dilution, W, S_old,  current_overlap, ext_input))) ./ p.tau
    return S_new
end 

function compute_noise_distribution(p, g, p_mat, overlap)
    sum = zeros(p.N)
    
    for j in 1:p.N
        s = 0.
        for i in p.P
            if i!=1 && i!=2
                ## check the prefactor (A * rm/N)!!!
                s = s + (p.A * g.rm) * ((p_mat[j, i] - p.γ) * overlap[i]) 
            end 

        end
        sum[j] = copy(s)
    end 
    return sum
end

function sigma_estimation(p, g, p_mat, overlap)
    alpha=p.P/p.N
    sum=0
    for i in 1:size(overlap)[1]
        if i!=1 && i!=2
            sum = sum +  overlap[i]^2
        end 
    end
    sigma = sqrt(p.γ*(1-p.γ)*sum)
    return sigma
end

function  evolve(; N = 10000, γ = 0.002, P = 10000, n = 2, t_max = 15, t_onset = 0.5, t_offset = 5.5, dt = 0.1,  I1 = 1, tau_s = 1,  rm = 1., b = 100, h0 = 0.25, A = 1., C_hat = 0, dilution = 0.006)
    # initialize the system state
    p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, tau = tau_s)
    g = Gain( rm = rm, b = b, h0 = h0)

    # Choose one out of the following two possible functions to generate the patterns
    p_mat = generate_binary_patterns_fix_sparseness(p) 
    #p_mat = generate_binary_patterns(p)
    start = time()
    W = weight_matrix(p, p_mat) #, dilution) 
    elapsed = time() - start
    println("----- weights are constructed -------- ", elapsed, " seconds")
    #c_mat = zeros(N,N)
    #W = weight_matrix(p, p_mat)
    S = 0. * p_mat[:,1]  
    ext1=  I1 .*p_mat[:,1] 
    ext2=  0.1 .*p_mat[:,2] 
    overlap_vect = Float64[]
    t_onset2 = 8
    t_offset2 = 13

    t = 0.
    open("Output_full_sim/overlap_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_rm$(g.rm)_b$(b).dat","w") do file
        #open("Output_full_sim/states_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_rm$(g.rm)_b$(b).dat","w") do s_t
        std = 0.
        overlap_vect = overlap(p, p_mat, S)
        write(file, "$(t)   ")
        write(file, "0  ")
        for i in 1 : p.P
            write(file, "$(overlap_vect[i])  ")
        end
        write(file, "\n ")
        while t < t_max
            sigma = sigma_estimation(p, g, p_mat, overlap_vect)
            #S = update_state_Euler(p, g, dt, p_mat, S, overlap_vect, ext)
            #println("current time = ", t)
            if t >= t_onset && t <= t_offset
                S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, ext1)
                #println("stim = ", 1)
            elseif t >= t_onset2 && t <= t_offset2
                S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, ext2)
            else
                S = update_state_Euler(p, g, dt, p_mat,  dilution, W, S, overlap_vect, 0.)
                #println("stim = ", 0)
            end
            overlap_vect = overlap(p, p_mat, S)
            t += dt
            write(file, "$(t)   ")
            write(file, "$(sigma)  ")
            for i in 1 : p.P
                write(file, "$(overlap_vect[i])  ")
            end
            write(file, "\n ")
            #write(s_t, "$(t)   ")
            #for i in 1 : p.N
            #    write(s_t, "$(S[i])  ")
            #end
            #write(s_t, "\n ")
        end
        #if t >= (t_max-2*dt)
        #open("Output_full_sim/final_noise_ditr_N$(N)_gamma$(γ)_P$(P)_tau$(tau_s)_dt$(dt)_chat$(C_hat)_A$(A)_rm$(rm)_b$(b).dat","w") do noise_ditr
        #    noise_ditr_vect = compute_noise_distribution(p, g, p_mat, overlap_vect)
        #    for i in 1:p.N
        #        write(noise_ditr, "$(noise_ditr_vect[i])  \n")
        #    end
        #end
        #end
    end 
    #open("Output_full_sim/state_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_rm$(g.rm)_b$(b).dat","w") do state
    #    for i in 1:p.N
    #        write(state, "$(S[i])  \n")
    #    end
    #end
    #println("state = ", done)
    #open("Output_full_sim/final_noise_ditr_N$(N)_gamma$(γ)_P$(P)_tau$(tau_s)_dt$(dt)_chat$(C_hat)_A$(A)_rm$(rm)_b$(b).dat","w") do noise_ditr
    #    noise_ditr_vect = compute_noise_distribution(p, g, p_mat, overlap_vect)
    #    for i in 1:p.N
    #        write(noise_ditr, "$(noise_ditr_vect[i])  \n")
    #    end
    #end
    elapsed = time() - start
    println("----- simulation time -------- ", elapsed, " seconds")
end
function experiment_proposal_evolve(; N = 10000, γ = 0.002, P = 10000, n = 2, t_max = 15, dt = 0.1, tau_s = 1,  rm = 1., b = 100, h0 = 0.25, A = 1., C_hat = 0, dilution = 0.006)
    p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, tau = tau_s)
    g = Gain( rm = rm, b = b, h0 = h0)
    p_mat = generate_random_binary_patterns_2contex_3people(p) 
    println("activity pattern 1 = ", sum(p_mat[:, 1]))
    println("activity pattern 2 = ", sum(p_mat[:, 2]))
    println("activity pattern 3 = ", sum(p_mat[:, 3]))
    println("activity pattern 4 = ", sum(p_mat[:, 4]))
    println("activity pattern 5 = ", sum(p_mat[:, 5]))
    println("corr patt 1 and 2 = ", sum(p_mat[:, 1] .* p_mat[:, 2]))
    println("corr patt 1 and 3 = ", sum(p_mat[:, 1] .* p_mat[:, 3]))
    println("corr patt 1 and 4 = ", sum(p_mat[:, 1] .* p_mat[:, 4]))
    println("corr patt 1 and 5 = ", sum(p_mat[:, 1] .* p_mat[:, 5]))
    println("corr patt 2 and 3 = ", sum(p_mat[:, 2] .* p_mat[:, 3]))
    println("corr patt 2 and 4 = ", sum(p_mat[:, 2] .* p_mat[:, 4]))
    println("corr patt 2 and 5 = ", sum(p_mat[:, 2] .* p_mat[:, 5]))
    println("corr patt 3 and 4 = ", sum(p_mat[:, 3] .* p_mat[:, 4]))
    println("corr patt 3 and 5 = ", sum(p_mat[:, 3] .* p_mat[:, 5]))
    println("corr patt 4 and 5 = ", sum(p_mat[:, 4] .* p_mat[:, 5]))
    
    
    start = time()
    W = weight_matrix(p, p_mat) #, dilution) 
    elapsed = time() - start
    println("----- weights are constructed -------- ", elapsed, " seconds")
    #c_mat = zeros(N,N)
    #W = weight_matrix(p, p_mat)
    S = rm .* p_mat[:,4]  
    overlap_vect = Float64[]
    ext = 0.08 .* (p_mat[:,2] .+ p_mat[:,3])
    t = 0.
    open("Output_full_sim/Exp_proposed_overlap_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_rm$(g.rm)_b$(b).dat","w") do file
        
        std = 0.
        overlap_vect = overlap(p, p_mat, S)
        write(file, "$(t)   ")
        write(file, "0  ")
        for i in 1 : p.P
            write(file, "$(overlap_vect[i])  ")
        end
        write(file, "\n ")
        while t < 2
            sigma = sigma_estimation(p, g, p_mat, overlap_vect)
            S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, 0.)
            overlap_vect = overlap(p, p_mat, S)
            t += dt
            write(file, "$(t)   ")
            write(file, "$(sigma)  ")
            for i in 1 : p.P
                write(file, "$(overlap_vect[i])  ")
            end
            write(file, "\n ")
        end
        while t >= 2 && t < 4
            sigma = sigma_estimation(p, g, p_mat, overlap_vect)
            S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, 0.)
            overlap_vect = overlap(p, p_mat, S)
            t += dt
            write(file, "$(t)   ")
            write(file, "$(sigma)  ")
            for i in 1 : p.P
                write(file, "$(overlap_vect[i])  ")
            end
            write(file, "\n ")
        end
        S = 0 .* p_mat[:,4]  
        while t >= 4 && t < 5
            sigma = sigma_estimation(p, g, p_mat, overlap_vect)
            S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, 0.)
            overlap_vect = overlap(p, p_mat, S)
            t += dt
            write(file, "$(t)   ")
            write(file, "$(sigma)  ")
            for i in 1 : p.P
                write(file, "$(overlap_vect[i])  ")
            end
            write(file, "\n ")
        end
        while t >= 5 && t < t_max
            sigma = sigma_estimation(p, g, p_mat, overlap_vect)
            
            if t >= 5 && t <= 6
                ext1=  rm .*p_mat[:,1]  
                S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, ext1 + ext)
                
            elseif t >= 10 && t <= 11
                ext2=  rm .*p_mat[:,4] 
                S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, ext2 + ext)
            else
                S = update_state_Euler(p, g, dt, p_mat,  dilution, W, S, overlap_vect, 0.)
                
            end
            overlap_vect = overlap(p, p_mat, S)
            t += dt
            write(file, "$(t)   ")
            write(file, "$(sigma)  ")
            for i in 1 : p.P
                write(file, "$(overlap_vect[i])  ")
            end
            write(file, "\n ")
            
        end
        
    end 
    
    elapsed = time() - start
    println("----- simulation time -------- ", elapsed, " seconds")
end
function experiment_proposal_evolve1(; N = 10000, γ = 0.002, P = 10000, n = 2, t_max = 15, dt = 0.1, tau_s = 1,  rm = 1., b = 100, h0 = 0.25, A = 1., C_hat = 0, dilution = 0.006)
    p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, tau = tau_s)
    g = Gain( rm = rm, b = b, h0 = h0)
    p_mat = generate_random_binary_patterns_2contex_3people(p) 
    println("activity pattern 1 = ", sum(p_mat[:, 1]))
    println("activity pattern 2 = ", sum(p_mat[:, 2]))
    println("activity pattern 3 = ", sum(p_mat[:, 3]))
    println("activity pattern 4 = ", sum(p_mat[:, 4]))
    println("activity pattern 5 = ", sum(p_mat[:, 5]))
    println("corr patt 1 and 2 = ", sum(p_mat[:, 1] .* p_mat[:, 2]))
    println("corr patt 1 and 3 = ", sum(p_mat[:, 1] .* p_mat[:, 3]))
    println("corr patt 1 and 4 = ", sum(p_mat[:, 1] .* p_mat[:, 4]))
    println("corr patt 1 and 5 = ", sum(p_mat[:, 1] .* p_mat[:, 5]))
    println("corr patt 2 and 3 = ", sum(p_mat[:, 2] .* p_mat[:, 3]))
    println("corr patt 2 and 4 = ", sum(p_mat[:, 2] .* p_mat[:, 4]))
    println("corr patt 2 and 5 = ", sum(p_mat[:, 2] .* p_mat[:, 5]))
    println("corr patt 3 and 4 = ", sum(p_mat[:, 3] .* p_mat[:, 4]))
    println("corr patt 3 and 5 = ", sum(p_mat[:, 3] .* p_mat[:, 5]))
    println("corr patt 4 and 5 = ", sum(p_mat[:, 4] .* p_mat[:, 5]))
    
    
    start = time()
    W = weight_matrix(p, p_mat) #, dilution) 
    elapsed = time() - start
    println("----- weights are constructed -------- ", elapsed, " seconds")
    #c_mat = zeros(N,N)
    #W = weight_matrix(p, p_mat)
    S = 0 .* p_mat[:,1]  
    overlap_vect = Float64[]
    ext = 0.02 .* (p_mat[:,2] .+ p_mat[:,3])
    t = 0.
    open("Output_full_sim/Exp_proposed_overlap_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_rm$(g.rm)_b$(b).dat","w") do file
        
        std = 0.
        overlap_vect = overlap(p, p_mat, S)
        write(file, "$(t)   ")
        write(file, "0  ")
        for i in 1 : p.P
            write(file, "$(overlap_vect[i])  ")
        end
        write(file, "\n ")
        while  t < t_max
            sigma = sigma_estimation(p, g, p_mat, overlap_vect)
            if t >= 0.99 && t <= 2
                ext1=  rm .*p_mat[:,1]  
                S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, ext1 + ext) 
            elseif t >= 4 && t <= 5
                ext2=  rm .*p_mat[:,5] 
                S = update_state_Euler(p, g, dt, p_mat,   dilution, W, S, overlap_vect, ext2 + ext)
            else
                S = update_state_Euler(p, g, dt, p_mat,  dilution, W, S, overlap_vect, ext)
            end
            overlap_vect = overlap(p, p_mat, S)
            t += dt
            write(file, "$(t)   ")
            write(file, "$(sigma)  ")
            for i in 1 : p.P
                write(file, "$(overlap_vect[i])  ")
            end
            write(file, "\n ")
            
        end
        
    end 
    
    elapsed = time() - start
    println("----- simulation time -------- ", elapsed, " seconds")
end


end # Module FullSim

module FullSim_DitributedGains
import ..AttractorNetwork: get_α_β , Heaviside, generate_binary_patterns, generate_binary_patterns_fix_sparseness, weight_matrix, overlap
using StatsBase
using LinearAlgebra
using Statistics
using Distributions
using Random

struct NetworkParameters
    N::Int64
    P::Int64
    n::Int64  ## number of correlated signals
    γ::Float64
    C_hat::Float64
    A::Float64
    α::Float64
    β::Float64
    tau::Float64
end
function NetworkParameters(; N, P, n, γ = 1/1000, A = 1, C_hat, tau = 1)   
    α, β = get_α_β(γ, C_hat)
    NetworkParameters(N, P, n, γ, C_hat, A, α, β, tau)
end

struct Gain
    b::Float64 
    h0::Float64
    rmin::Float64
    rmax::Float64
end
function Gain(; b = 0.8, h0 = 10., rmin = 0., rmax = 50.)
    Gain(b, h0, rmin, rmax)
end
(g::Gain)(x) = (g.rmax - g.rmin) ./ (1 .+ exp.(- g.b .* (x .- g.h0))) .+  g.rmin

function GenerateGains(p, μ_min, σ_min, μ_max, σ_max ,b, h0)
    gains = []
    drmax = Normal( μ_max,  σ_max )
    drmin = Normal( μ_min,  σ_max )
    for i in 1:p.N
        rmax = max( 0.5, h0, rand(drmax,1)[1] )
        rmin = max( 0, rand(drmin,1)[1] )
        h1 = max(h0 * (rmax - rmin), 0.25)
        g = Gain( b = b, h0 = h1, rmin = rmin, rmax = rmax)
        push!(gains, g)
    end
    return gains
end 
"""
function  overlap(p, pattern_matrix, S, gains)  
    normalization = 1 /(p.N * p.γ * (1 - p.γ))
    overlaps = zeros(p.P)
    #println(max(0., (S[10]- gains[10].rmin)/(gains[10].rmax- gains[10].rmin)),"   ", (S[10]- gains[10].rmin)/(gains[10].rmax- gains[10].rmin))
    #println((S[10]- gains[10].rmin)/(gains[10].rmax- gains[10].rmin))
    println(S[10], "   ", S[10]/gains[10].rmax)
    for mu in 1:p.P
        sum = 0.
        for i in 1:p.N
            #overlaps[:] += normalization * ((pattern_matrix[i, :] .- p.γ) .* ((S[i]- gains[i].rmin)/(gains[i].rmax- gains[i].rmin)))
            sum += (pattern_matrix[i, mu] - p.γ) * (S[i]/gains[i].rmax)  #max(0, ((S[i]- gains[i].rmin)/(gains[i].rmax- gains[i].rmin)))
        end
        overlaps[mu] = normalization * sum
    end
    return overlaps
end

function  update_state_Euler(p, W, gains, h, p_mat, S, current_overlap, ext_input, μ_max)
    #println("udating euler")
    S_old = copy(S)
    current_overlap_old = copy(current_overlap)
    S_new = zeros(p.N)
    println("udating euler")
    for i in 1:p.N
        #input = (p.A * μ_max * (p_mat[i,:] .- p.γ)' * current_overlap) + ext_input[i]
        sum = 0
        for j in 1:p.N
            sum += W[i,j] * max(0., (S_old[j]- gains[j].rmin)/(gains[j].rmax- gains[j].rmin)) #S[j]
        end
        input = sum + ext_input[i]
        phi = gains[i](input)    # [g(input) for g in gains]  #
        S_new[i] = S_old[i] + h * (- S_old[i] + phi) / p.tau
    end 
    return S_new
end 
"""
function  update_state_Euler(p, W, gains, h, p_mat, S, current_overlap, ext_input, μ_max) # quella buona
    #println("udating euler")
    S_old = copy(S)
    current_overlap_old = copy(current_overlap)
    S_new = zeros(p.N)
    println("updating euler")
    sum = zeros(p.N)
    for j in 1:p.N
        sum[:] += W[:,j] * max(0., (S_old[j]- gains[j].rmin) /(gains[j].rmax- gains[j].rmin)) #S[j] #
    end
    
    input = sum .+ ext_input
    phi = Float64[]
    for i in 1:p.N
        push!(phi, gains[i](input[i]) )   # [g(input) for g in gains]  #
    end
    S_new = S_old .+ h .* (- S_old .+ phi) ./ p.tau
    return S_new
end 

function sigma_estimation(p, p_mat, overlap)
    alpha=p.P/p.N
    sum=0
    for i in 1:size(overlap)[1]
        if i!=1 && i!=2
            sum = sum +  overlap[i]^2
        end 
    end
    sigma = sqrt(p.γ*(1-p.γ)*sum)
    return sigma
end

function  evolve(; N = 10000, γ = 0.002, P = 1000, n = 2, t_max = 15, t_onset = 0.5, t_offset = 8, dt = 0.1,  I1 = 1, tau_s = 1,  b = 100.,  A = 1., C_hat = 0., h0 = 0.4, μ_min = 0.25, σ_min = 0.1, μ_max = 1., σ_max = 0.35)
    # initialize the system state
    p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, tau = tau_s)
    # GenerateGains(p, μ_min, σ_min, μ_max, σ_max ,b, h0)
    println("h0 = ", h0)
    gains = GenerateGains(p, μ_min, σ_min, μ_max, σ_max, b, h0)
    println("gains[1](0) = ", gains[1] )
    println("gains[1](0) = ", gains[1](0) )

    # Choose one out of the following two possible functions to generate the patterns
    p_mat = generate_binary_patterns_fix_sparseness(p) 
    #p_mat = generate_binary_patterns(p)
    
    W = weight_matrix(p, p_mat)
    S = 0. * p_mat[:,1]  
    ext=  I1 .*p_mat[:,1] 
    overlap_vect = Float64[] 
    #overlap_vect = overlap(p, p_mat, S)
    stim = 0. * p_mat[:,1]  

    t = 0.
    open("Output_full_sim_distr_gains/overlap_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_b$(b).dat","w") do over
        std = 0.
        #write(over, "$(t)   ")
        #write(over, "0  ")
        #for i in 1 : p.P
        #    write(over, "$(overlap_vect[i])  ")
        #end
        #write(over, "\n ")
        while t < t_max
            sigma = sigma_estimation(p, p_mat, overlap_vect)
            if t >= t_onset && t <= t_offset
                #                       p, gains, h, p_mat, S, current_overlap, ext_input, μ_max
                S = update_state_Euler(p, W, gains, dt, p_mat, S, overlap_vect, ext, μ_max)
                
                #println("external active units = ",sum(ext))
            else
                S = update_state_Euler(p, W, gains, dt, p_mat, S, overlap_vect, stim, μ_max)
                
                #println("external active units = ",sum(stim))
            end
            overlap_vect = overlap(p, p_mat, S) #, gains)
            t += dt
            write(over, "$(t)   ")
            write(over, "$(sigma)  ")
            for i in 1 : p.P
                write(over, "$(overlap_vect[i])  ")
            end
            write(over, "\n ")
            
        end
    end 
end

end # Module FullSim_DitributedGains


module AdaptFullSim
import ..AttractorNetwork: get_α_β , Gain, Heaviside, generate_binary_patterns, generate_binary_patterns_fix_sparseness, weight_matrix, input_field, overlap, generate_random_binary_patterns_single_group, remove!, generate_parent_binary_patterns_fix_sparseness_single_group, generate_strict_parent_binary_patterns_single_group, generate_2groups_binary_patterns_fix_sparseness
using StatsBase
using LinearAlgebra
using Statistics
using Distributions
using Random
using Waveforms

struct NetworkParameters
    N::Int64
    P::Int64
    n::Int64  ## number of correlated signals
    γ::Float64
    C_hat::Float64
    A::Float64
    α::Float64
    β::Float64
    J0::Float64
end
function NetworkParameters(; N, P, n, γ = 1/1000, A = 1, C_hat, J0)   
    α, β = get_α_β(γ, C_hat)
    NetworkParameters(N, P, n, γ, C_hat, A, α, β, J0)
end

function  feed_back_inh(p, S)
    #S = state of the network, N n_active_units
    return p.J0 / (p.γ * p.N) * sum(S)
end 
function  update_state_Euler(p, g, S, W, tau, h, ext_input, thr, thr_0, Dth, Tth, p_mat, current_overlap)
    S = S .+ h.*( - S .+ g(input_field(p, g, p_mat, current_overlap, ext_input) .- feed_back_inh( p, S) .- thr)) ./tau
    thr = thr .- h.*(thr .- thr_0 .- (Dth .* S)) ./ Tth
    return S, thr
end 
function  evolve(; N = 3000, γ = 0.1, P = 16, n = 2, t_max = 1000, t_onset = 0, t_offset = 0, dt = 1,  I1 = 0, tau_s = 1,  rm = 1, b = 2000, h0 = 0, A = 1, C_hat = 0, T=0.015, Tth=45, TJ0=25, min_J0=0.7, max_J0 = 1.2)
    Dth = 1.9 * T
    # initialize the system state
    p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, J0 = 0)
    g = Gain( rm = rm, b = b, h0 = h0)
    #g = 0
    thresh_0 = rand(Uniform(h0 - T, h0 + T), p.N) 
    thr = copy(thresh_0)
    p1 = NetworkParameters(N = N, P = 12, n = 12, γ = γ , A = A, C_hat = C_hat, J0 = 0)
    p2 = NetworkParameters(N = N, P = 4, n = 4, γ = γ , A = A, C_hat = C_hat, J0 = 0)
    p_mat =  generate_2groups_binary_patterns_fix_sparseness(p, p1, p2)  # generate_binary_patterns_fix_sparseness(p)  #

    println("activity pattern 1 = ", sum(p_mat[:, 1]))
    println("activity pattern 2 = ", sum(p_mat[:, 2]))
    println("activity pattern 3 = ", sum(p_mat[:, 3]))
    println("activity pattern 4 = ", sum(p_mat[:, 4]))
    println("activity pattern 5 = ", sum(p_mat[:, 5]))
    println("activity pattern 6 = ", sum(p_mat[:, 6]))
    println("activity pattern 7 = ", sum(p_mat[:, 7]))
    println("activity pattern 8 = ", sum(p_mat[:, 8]))
    println("activity pattern 9 = ", sum(p_mat[:, 9]))
    println("activity pattern 10 = ", sum(p_mat[:, 10]))
    println("activity pattern 11 = ", sum(p_mat[:, 11]))
    println("activity pattern 12 = ", sum(p_mat[:, 12]))
    println("activity pattern 13 = ", sum(p_mat[:, 13]))
    println("activity pattern 14 = ", sum(p_mat[:, 14]))
    println("activity pattern 15 = ", sum(p_mat[:, 15]))
    println("activity pattern 16 = ", sum(p_mat[:, 16]))
    println("corr patt 1 and 2 = ", sum(p_mat[:, 1] .* p_mat[:, 2]))
    println("corr patt 1 and 3 = ", sum(p_mat[:, 1] .* p_mat[:, 3]))
    println("corr patt 1 and 4 = ", sum(p_mat[:, 1] .* p_mat[:, 4]))
    println("corr patt 2 and 4 = ", sum(p_mat[:, 2] .* p_mat[:, 4]))
    println("corr patt 3 and 4 = ", sum(p_mat[:, 3] .* p_mat[:, 4]))
    println("corr patt 1 and 5 = ", sum(p_mat[:, 1] .* p_mat[:, 5]))
    println("corr patt 1 and 6 = ", sum(p_mat[:, 1] .* p_mat[:, 6]))
    println("corr patt 1 and 7 = ", sum(p_mat[:, 1] .* p_mat[:, 7]))
    println("corr patt 1 and 8 = ", sum(p_mat[:, 1] .* p_mat[:, 8]))
    println("corr patt 1 and 9 = ", sum(p_mat[:, 1] .* p_mat[:, 9]))
    println("corr patt 1 and 10 = ", sum(p_mat[:, 1] .* p_mat[:, 10]))
    println("corr patt 1 and 11 = ", sum(p_mat[:, 1] .* p_mat[:, 11]))
    println("corr patt 1 and 12 = ", sum(p_mat[:, 1] .* p_mat[:, 12]))
    println("corr patt 1 and 13 = ", sum(p_mat[:, 1] .* p_mat[:, 13]))
    println("corr patt 1 and 14 = ", sum(p_mat[:, 1] .* p_mat[:, 14]))
    println("corr patt 1 and 15 = ", sum(p_mat[:, 1] .* p_mat[:, 15]))
    println("corr patt 1 and 16 = ", sum(p_mat[:, 1] .* p_mat[:, 16]))
    println("corr patt 2 and 16 = ", sum(p_mat[:, 2] .* p_mat[:, 16]))
    println("corr patt 14 and 16 = ", sum(p_mat[:, 14] .* p_mat[:, 16]))
    println("corr patt 15 and 16 = ", sum(p_mat[:, 15] .* p_mat[:, 16]))
    println("corr patt 14 and 15 = ", sum(p_mat[:, 14] .* p_mat[:, 15]))
    println("corr patt 10 and 16 = ", sum(p_mat[:, 10] .* p_mat[:, 16]))
    
    W = weight_matrix(p, p_mat)
    S = 0. .* p_mat[:,1] #rm .* p_mat[:,1]  
    #ext=  I1 .*p_mat[:,1] 
    println("----------- so far so good -----------------")
    t = 0.
    open("Output_full_sim_adapt/adapt_overlap_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_rm$(g.rm)_b$(b).dat","w") do overlap_file
        std=0.
        while t < t_max
            overlap_vect = overlap(p, p_mat, S)
            J0 = (max_J0 - min_J0) / 2 * sin(t * (2 * pi) / TJ0 - pi/2)+(max_J0 + min_J0) / 2           # Sinusoidal periodic inhibition
            #J0 =  (max_J0-min_J0) / 2. * squarewave(2 * pi * t/ TJ0 - pi/2.)+ (max_J0 + min_J0) / 2.   # Squared wave periodic inhibition
            p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, J0 = J0)
            if t > 1 && t < 10
                ext=  rm/15 .*p_mat[:,1] 
                S, thr = update_state_Euler(p, g, S, W, tau_s, dt, ext, thr, thresh_0, Dth, Tth, p_mat, overlap_vect )
            elseif t > 190 && t < 200
                ext=  rm/15 .*p_mat[:,3] 
                S, thr = update_state_Euler(p, g, S, W, tau_s, dt, ext, thr, thresh_0, Dth, Tth, p_mat, overlap_vect )
            elseif t > 390 && t < 400
                ext=  rm .*p_mat[:,15] 
                S, thr = update_state_Euler(p, g, S, W, tau_s, dt, ext, thr, thresh_0, Dth, Tth, p_mat, overlap_vect )
            else
                ext=  0. .*p_mat[:,1] 
                S, thr = update_state_Euler(p, g, S, W, tau_s, dt, ext, thr, thresh_0, Dth, Tth, p_mat, overlap_vect )
            end
            t += dt
            write(overlap_file, "$(t)   ")
            for i in 1 : p.P
                write(overlap_file, "$(overlap_vect[i])  ")
            end
            write(overlap_file, "$(J0)  ")
            write(overlap_file, "\n ")
        end
    end 
end


end # end of module AdaptFullSim

module AdaptFullSimDistrGains
import ..AttractorNetwork: get_α_β , generate_binary_patterns, generate_binary_patterns_fix_sparseness, weight_matrix, input_field, overlap
using StatsBase
using LinearAlgebra
using Statistics
using Distributions
using Random
using Waveforms

struct NetworkParameters
    N::Int64
    P::Int64
    n::Int64  ## number of correlated signals
    γ::Float64
    C_hat::Float64
    A::Float64
    α::Float64
    β::Float64
    J0::Float64
end
function NetworkParameters(; N, P, n, γ = 1/1000, A = 1, C_hat, J0)   
    α, β = get_α_β(γ, C_hat)
    NetworkParameters(N, P, n, γ, C_hat, A, α, β, J0)
end

struct Gain
    b::Float64 
    h0::Float64
    rmin::Float64
    rm::Float64
end
function Gain(; b = 0.8, h0 = 10., rmin = 0., rm = 50.)
    Gain(b, h0, rmin, rm)
end
(g::Gain)(x) = (g.rm - g.rmin) ./ (1 .+ exp.(- g.b .* (x .- g.h0))) .+  g.rmin

function GenerateGains(p, μ_min, σ_min, μ_max, σ_max ,b, h0)
    gains = []
    drmax = Normal( μ_max,  σ_max )
    drmin = Normal( μ_min,  σ_max )
    for i in 1:p.N
        rmax = max( h0, rand(drmax,1)[1] )
        rmin = max( 0, rand(drmin,1)[1] )
        h1 = h0 * (rmax - rmin)
        g = Gain( b = b, h0 = h1, rmin = rmin, rm = rmax)
        push!(gains, g)
    end
    return gains
end 

function  feed_back_inh(p, S)
    #S = state of the network, N n_active_units
    return p.J0 / (p.γ * p.N) * sum(S)
end 
function  update_state_Euler(p, g, S, W, tau, h, ext_input, thr, thr_0, Dth, Tth, p_mat, current_overlap, gains)
    S_old = copy(S)
    current_overlap_old = copy(current_overlap)
    S_new = zeros(p.N)
    input = zeros(p.N)
    #for i in 1:p.N
    #    for j in 1:p.N
    #        input[i] = W[ij] * max(0., (S_old[j]- gains[j].rmin) /(gains[j].rmax- gains[j].rmin)) #input_field(p, g, p_mat, current_overlap_old, ext_input)
    #    end
    #end
    #feedback = 0.
    S_temp = copy(S)
    for i in 1:p.N
        S_temp[i] = max(0.,(S_old[i] - gains[i].rmin) /(gains[i].rm- gains[i].rmin)) #
        #feedback += S_temp[i] #S_old[i]/ gains[i].rm
    end
    feedback = p.J0 / (p.γ * p.N) * sum(S_temp)
    input = W*S_temp
    for i in 1:p.N
        #input = (p.A * μ_max * (p_mat[i,:] .- p.γ)' * current_overlap) + ext_input[i]    #input_field(p, g, p_mat, current_overlap_old, ext_input)
        #println("input = ", size(input))
        S_new[i] = S_old[i] + h *( - S_old[i] + gains[i](input[i] -  feedback - thr[i]))/tau
    end 
    thr = thr .- h.*(thr .- thr_0 .- (Dth .* S)) ./ Tth
    return S_new, thr
end 
function  evolve(; N = 3000, γ = 0., P = 16, n = 2, t_max = 1000, t_onset = 0, t_offset = 0, dt = 1,  I1 = 0, tau_s = 1,  μ_min = 0.1, σ_min = 0.1, μ_max = 1., σ_max = 0.1, b = 1000., h0 = 0, A = 1, C_hat = 0, T=0.015, Tth=45, TJ0=25, min_J0=0.7, max_J0 = 1.2)
    Dth = 1.9 * T
    # initialize the system state
    p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, J0 = 0)
    gains = GenerateGains(p, μ_min, σ_min, μ_max, σ_max, b, h0)
    #println(gains[1], gains[2])
    g = Gain( b = b, h0 = h0, rm = μ_max, rmin = μ_min)
    thresh_0 = rand(Uniform(h0 - T, h0 + T), p.N) 
    thr = copy(thresh_0)

    # Choose one out of the following two possible functions to generate the patterns
    p_mat = generate_binary_patterns_fix_sparseness(p) 
    println("size pmat =", size(p_mat))
    #p_mat = generate_binary_patterns(p)

    println("activity pattern 1 = ", sum(p_mat[:, 1]))
    println("activity pattern 2 = ", sum(p_mat[:, 2]))
    println("activity pattern 3 = ", sum(p_mat[:, 3]))
    println("corr patt 1 and 2 = ", sum(p_mat[:, 1] .* p_mat[:, 2]))
    println("corr patt 1 and 3 = ", sum(p_mat[:, 1] .* p_mat[:, 3]))
    println("corr patt 2 and 3 = ", sum(p_mat[:, 2] .* p_mat[:, 3]))
    println("corr patt 1 and 4 = ", sum(p_mat[:, 1] .* p_mat[:, 4]))
    
    W = weight_matrix(p, p_mat)
    S = μ_max .* p_mat[:,1]  
    ext=  I1 .*p_mat[:,1] 

    t = 0.
    open("Output_full_sim_adapt_distr_gains/adapt_overlap_time_neus$(p.N)_gamma$(p.γ)_P$(p.P)_tau$(tau_s)_dt$(dt)_chat$(p.C_hat)_A$(p.A)_rm$(g.rm)_b$(b).dat","w") do over
        std=0.
        while t < t_max
            overlap_vect = overlap(p, p_mat, S)
            J0 = (max_J0 - min_J0) / 2 * sin(t * (2 * pi) / TJ0 - pi/2)+(max_J0 + min_J0) / 2           # Sinusoidal periodic inhibition
            #J0 =  (max_J0-min_J0) / 2. * squarewave(2 * pi * t/ TJ0 - pi/2.)+ (max_J0 + min_J0) / 2.   # Squared wave periodic inhibition
            p = NetworkParameters(N = N, P = P, n = n, γ = γ , A = A, C_hat = C_hat, J0 = J0)
            S, thr = update_state_Euler(p, g, S, W, tau_s, dt, ext, thr, thresh_0, Dth, Tth, p_mat, overlap_vect, gains )
            t += dt
            write(over, "$(t)   ")
            for i in 1 : p.P
                write(over, "$(overlap_vect[i])  ")
            end
            write(over, "$(J0)  ")
            write(over, "\n ")
        end
    end 
end
end # end of module AdaptFullSimDistrGains

end # end of module AttractorNetwork