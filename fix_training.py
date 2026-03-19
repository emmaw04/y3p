import re

# Update src/training/evaluate_base.py
with open("src/training/evaluate_base.py", "r") as f:
    eval_base = f.read()

# Replace any lingering mentions or add the new model to seq_models set
if 'seq_models = {"tcn", "gru", "lstm", "tcn_gru"}' in eval_base:
    eval_base = eval_base.replace(
        'seq_models = {"tcn", "gru", "lstm", "tcn_gru"}',
        'seq_models = {"tcn", "gru", "lstm", "tcn_gru", "hybrid_vse"}'
    )
    
with open("src/training/evaluate_base.py", "w") as f:
    f.write(eval_base)

# Update src/models/models.py to register it
with open("src/models/models.py", "r") as f:
    models_py = f.read()

if '"hybrid_vse": make_hybrid_vse_binary(cfg),' not in models_py:
    old_reg = '''def get_stage1_sequential_models(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """registry for the stage 1 sequence models"""

    return {
        "tcn": make_tcn_binary(cfg),
        "tcn_gru": make_tcn_gru_binary(cfg),
        "lstm": make_lstm_binary(cfg),
        "gru": make_gru_binary(cfg),
    }'''

    new_reg = '''def get_stage1_sequential_models(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """registry for the stage 1 sequence models"""

    return {
        "tcn": make_tcn_binary(cfg),
        "tcn_gru": make_tcn_gru_binary(cfg),
        "lstm": make_lstm_binary(cfg),
        "gru": make_gru_binary(cfg),
        "hybrid_vse": make_hybrid_vse_binary(cfg),
    }'''

    models_py = models_py.replace(old_reg, new_reg)
    with open("src/models/models.py", "w") as f:
        f.write(models_py)

