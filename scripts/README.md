
### Steps to reproduce HumanEval result

```
python scripts/eval.py --base_model budecosystem/code-millenials-34b --dataset humaneval

evalplus.evaluate --dataset humaneval --samples $SAVE_PATH
```