import threading
import time
import weakref
import torch
import logging
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelCache:
    def __init__(self, timeout_minutes: int = 5):
        self.timeout_minutes = timeout_minutes
        self.models = {}
        self.last_used = {}
        self.lock = threading.Lock()
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        def cleanup():
            while True:
                time.sleep(60)  # Check every minute
                self._cleanup_expired_models()
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_expired_models(self):
        current_time = datetime.now()
        with self.lock:
            for key in list(self.models.keys()):
                if key in self.last_used:
                    elapsed = (current_time - self.last_used[key]).total_seconds() / 60
                    if elapsed > self.timeout_minutes:
                        logger.info(f"Unloading model {key} due to inactivity")
                        model = self.models.pop(key)
                        self.last_used.pop(key)
                        if hasattr(model, 'to'):
                            model.to('cpu')
                        del model
                        torch.cuda.empty_cache()

    def get_model(self, key: str, model_loader: callable) -> Any:
        with self.lock:
            if key not in self.models:
                logger.info(f"Loading model {key}")
                self.models[key] = model_loader()
            self.last_used[key] = datetime.now()
            return self.models[key]

# Global cache instance
model_cache = ModelCache(timeout_minutes=5)