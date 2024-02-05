(1) generate synthetic data-set: python get_data_synthetic.py "sl" or  python get_data_synthetic.py "asl"
(2) generate baseline data-set of financial returns: python baseline_data_gen.py 
(3) complete parameter files: mixture_asl.yaml,  mixture_sl.yaml, and yearly_max_params.yaml
(4) execute training, where "evd" / "unc" / "evd-sm" must be inputed and exectued for each "experiment" in the .yaml file: max_gen_baseline.py mixture_asl.yaml, max_gen_baseline.py mixture_sl.yaml, and max_gen_baseline.py yearly_max_params.yaml
(5) plot output: plot_dro_synthetic.py mixture_asl.yaml, plot_dro_synthetic.py mixture_sl.yaml, and plot_dro_baseline.py yearly_max_params.yaml
