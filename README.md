To train the model (e.g. TSP with 20 nodes)
```bash
python run.py --graph_size 20
```
To evaluate the test dataset with pretrained model for TSP20
```bash
python eval.py --model pretrained/tsp20/epoch-99.pt --datasets [data/tsp20_test_seed1234.pkl]
```

