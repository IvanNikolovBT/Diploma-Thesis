import optuna
from sentence_transformers import SentenceTransformer
from vectorbuilder import VectorDBBuilder
import logging
import sqlite3
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database to store Optuna trials
DB_PATH = "optuna_studies.db"
STUDY_NAME = "macedonian_vectordb_optimization"

def objective(trial):
    try:
        # Hyperparameters to test
        chunk_size = trial.suggest_int("chunk_size", 100, 1500, step=100)
        chunk_overlap = trial.suggest_int("chunk_overlap", 100, chunk_size // 2, step=50)
        model_name = trial.suggest_categorical("model_name", [
            "sentence-transformers/all-MiniLM-L6-v2",
            "paraphrase-multilingual-mpnet-base-v2",
            "BAAI/bge-m3",
            "distiluse-base-multilingual-cased-v2"
        ])

        logger.info(f"\nTrial {trial.number}: Testing chunk_size={chunk_size}, overlap={chunk_overlap}, model={model_name}")

        # Build and query
        builder = VectorDBBuilder(
            CHUNK_SIZE=chunk_size,
            CHUNK_OVERLAP=chunk_overlap,
            model_name=model_name
        )
        builder.build_database_fully()

        query = "Петре М. Андреевски песни за љубов"
        results = builder.query_database_semantic(query)
        
        if not results or len(results["distances"]) == 0:
            logger.warning("No results returned - penalizing trial")
            return 0.0
        
        avg_similarity = sum(float(d) for d in results["distances"]) / len(results["distances"])
        logger.info(f"Avg similarity: {avg_similarity:.4f}")
        
        return avg_similarity

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        return 0.0  # Return 0 for failed trials

def load_or_create_study():
    storage = f"sqlite:///{DB_PATH}"
    
    try:
        # Try to load existing study
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage
        )
        logger.info(f"Resuming existing study with {len(study.trials)} completed trials")
    except:
        # Create new study if none exists
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )
        logger.info("Created new study")
    
    return study

if __name__ == "__main__":
    study = load_or_create_study()
    
    # Run optimization (can be stopped and restarted)
    study.optimize(objective, n_trials=50, gc_after_trial=True)

    # Print best results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Avg Similarity: {trial.value:.4f}")
    print(f"  Parameters: {trial.params}")