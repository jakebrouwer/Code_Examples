# Code_Examples
Coding Examples to reflect generalized personal coding practices

The following shows the proper import path and how to run examples from the Example_Code directory:
>  from Example_Code import Master
> 
> master = Master()
> 
> master.self_powers()\
> master.find_dna_motif(dna="GATATATGCATATACTT",motif="ATAT")


The following shows the proper import path and how to run examples from the ML_Examples directory:
> from ML_Examples import ClassicalML,AutoEncoder

> classic = ClassicalML(data='breast_cancer')\
> classic.find_best_model()\
> classic.optimize_best_model()\
> classic.score_best_model()

> ae = AutoEncoder(data='mushroom')\
> ae.build_autoencoder()
