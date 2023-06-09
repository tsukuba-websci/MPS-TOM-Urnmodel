using PolyaUrnSimulator
using StatsBase
using ProgressMeter

include("../lib/history2vec.jl")

function run_existing_model(rho::Int, nu::Int, s::String)
  _s::Union{Function,Nothing} = nothing
  if s == "SSW"
    _s = ssw_strategy!
  elseif s == "WSW"
    _s = wsw_strategy!
  else
    throw(error("strategy must be SSW or WSW"))
  end

  env = Environment()
  init_agents = [
    Agent(rho, nu, _s)
    Agent(rho, nu, _s)
  ]
  init!(env, init_agents)


  for _ in 1:20000
    step!(env)
  end

  return history2vec(env.history, 1000)
end

function full_search()
  outdir = mkpath("results")
  outfile = "$outdir/existing_full_search.csv"

  open(outfile, "w") do fp
    println(fp, "rho,nu,s,gamma,no,nc,oo,oc,c,y,g,r,h")
  end

  N = 1:10
  rhos = 1:20
  nus = 1:20
  ss = ["SSW", "WSW"]

  p = Progress(length(N) * length(rhos) * length(nus) * length(ss))
  lk = ReentrantLock()
  Threads.@threads for _ in N
    Threads.@threads for rho in rhos
      Threads.@threads for nu in nus
        Threads.@threads for s in ss
          res = run_existing_model(rho, nu, s)
          lock(lk) do
            open(outfile, "a") do fp
              println(fp, join(string.(values((; rho, nu, s, res...))), ","))
            end
          end
          next!(p)
        end
      end
    end
  end
end

full_search()
