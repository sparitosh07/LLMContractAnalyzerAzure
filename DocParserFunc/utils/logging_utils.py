"""Comprehensive logging utilities similar to AzureML RAG."""

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Optional
from functools import wraps


class ActivityLogger:
    """Activity logger for tracking processing metrics."""
    
    def __init__(self, activity_name: str, logger: logging.Logger):
        """Initialize activity logger."""
        self.activity_name = activity_name
        self.logger = logger
        self.activity_info: Dict[str, Any] = {}
        self.start_time = time.time()
        
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.logger.info(f"[{self.activity_name}] {message}", extra=extra)
        
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message.""" 
        self.logger.error(f"[{self.activity_name}] {message}", extra=extra)
        
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.logger.warning(f"[{self.activity_name}] {message}", extra=extra)
        
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.logger.debug(f"[{self.activity_name}] {message}", extra=extra)
        
    def set_activity_info(self, key: str, value: Any):
        """Set activity information."""
        self.activity_info[key] = value
        
    def get_duration(self) -> float:
        """Get activity duration in seconds."""
        return time.time() - self.start_time
        
    def complete(self):
        """Mark activity as complete and log summary."""
        duration = self.get_duration()
        self.activity_info["duration_seconds"] = duration
        self.info(f"Activity completed in {duration:.2f} seconds")
        self.info(f"Activity info: {self.activity_info}")


@contextmanager
def track_activity(logger: logging.Logger, activity_name: str):
    """Context manager for tracking activity metrics."""
    activity_logger = ActivityLogger(activity_name, logger)
    activity_logger.info(f"Starting activity: {activity_name}")
    
    try:
        yield activity_logger
    except Exception as e:
        activity_logger.error(f"Activity failed with exception: {str(e)}")
        activity_logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        activity_logger.complete()


def safe_log_metric(
    metric_name: str, 
    value: Any, 
    logger: logging.Logger, 
    step: Optional[int] = None
):
    """Safely log a metric with error handling."""
    try:
        extra = {"metric_name": metric_name, "metric_value": value}
        if step is not None:
            extra["step"] = step
            
        logger.info(f"Metric - {metric_name}: {value}", extra=extra)
    except Exception as e:
        logger.warning(f"Failed to log metric {metric_name}: {str(e)}")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get configured logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        
    return logger


def log_processing_stats(
    logger: logging.Logger,
    stats: Dict[str, Any],
    activity_logger: Optional[ActivityLogger] = None
):
    """Log document processing statistics."""
    logger.info("Processing Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
        if activity_logger:
            activity_logger.set_activity_info(key, value)


def exception_handler(logger: logging.Logger):
    """Decorator for comprehensive exception handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        return wrapper
    return decorator


class LoggingConfig:
    """Centralized logging configuration."""
    
    @staticmethod
    def setup_logging(
        level: int = logging.INFO,
        format_string: Optional[str] = None,
        include_timestamp: bool = True
    ):
        """Setup global logging configuration."""
        if format_string is None:
            if include_timestamp:
                format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            else:
                format_string = '%(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=level,
            format=format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Set specific logger levels
        logging.getLogger('azure').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
    @staticmethod
    def get_function_logger(function_name: str) -> logging.Logger:
        """Get logger for Azure Function."""
        return get_logger(f"azure_function.{function_name}")


# Performance monitoring decorators
def monitor_performance(logger: logging.Logger, metric_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                safe_log_metric(f"{metric_name}_duration", duration, logger)
                safe_log_metric(f"{metric_name}_success", 1, logger)
                return result
            except Exception as e:
                duration = time.time() - start_time
                safe_log_metric(f"{metric_name}_duration", duration, logger)
                safe_log_metric(f"{metric_name}_failure", 1, logger)
                logger.error(f"Performance monitoring - {func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator