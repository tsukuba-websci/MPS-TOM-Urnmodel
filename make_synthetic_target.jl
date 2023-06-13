using PolyaUrnSimulator
using StatsBase
using ProgressMeter

include("lib/history2vec.jl")

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

function synthetic_target()
  outdir = mkpath("data")
  outfile = "$outdir/synthetic_target.csv"

  open(outfile, "w") do fp
    println(fp, "rho,nu,s,gamma,no,nc,oo,oc,c,y,g,r,h")
  end

  N = 1:10
  parameter_sets = [
    Dict("rho" => 5, "nu" => 15),
    Dict("rho" => 21, "nu" => 7),
    Dict("rho" => 5, "nu" => 5)
  ]
  ss = ["SSW", "WSW"]

  p = Progress(length(N) * length(parameter_sets) * length(ss))
  lk = ReentrantLock()
  Threads.@threads for _ in N
    Threads.@threads for parameter_set in parameter_sets
      Threads.@threads for s in ss
        rho = parameter_set["rho"]
        nu = parameter_set["nu"]
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

synthetic_target()