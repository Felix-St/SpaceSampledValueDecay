# Space-sampled Value Decay (SsVD): Forgetting Mechanisms for Non-stationary Reinforcement Learning



### Repository Overview
The following diagram shows how files are organized in this repository and how these can be used for reproducibility.

<pre>
src/
├── utils/
│   <b>└── model_configuration.py         # Contains configurations to run diff. benchmark settings </b>
│   └── plotting.py
│   
├── models/                            
│   <b>└── custom_agents.py               # Implementation for DQN_F, SAC_F</b>
│   └── baselines.py                   # Implementation for L2ToInit, PeriodicResets
│   └── boot_dqn.py                    # BootstrappedDQN w/ rdm. priors impl. for DeepSea 
│
├── environments/                      # Includes custom and external (e.g. DeepSea) environments
│   └── choose_box.py                  # Implementation for ChooseBox env
│   └── river_swim.py 				   # Implementation of the RiverSwim env
│   └── deep_sea.py                    # Implementation of the DeepSea env
│   └── nonstationary_lunar_lander.py  # Implementation of the MultiPadLunarLander 
│   └── nsrl_tracking_wrapper.py       # Wrapper to track when a ns-gym environment has update (i.e. drifted) 
│
├── results/                           # Application source code
│   └── *.json                         # Contains the actual raw results
│ 
<b>└── evaluate_nonstgym_parallel.py      # Script to run (paralellized) main eval (diff. benchmarks are configured with utils/model_configuration) </b>
└── evaluate_deep_sea.py               # Script to run the deep sea scaling experiment
│ 
└── evaluate_and_plot_visabs.py            # Runs and visualizes the vis. abstr., i.e. Fig. 1 and 10 with MountainCar
│ 
<b>└── plot_seosnsrl.py                   # Visualize main results (based on results/*.json) (i.e. Figure 2 and 9) </b> 
└── plot_hard_exploration_combined.py  # Combine exploration results of the classic eval + DeepSea (i.e. Figure 4) 
└── plot_ranking.py					   # Creates ranking results i.e. Figure 3
└── requirements.txt                   # Dependencies: pip install -r requirements.txt
</pre>
