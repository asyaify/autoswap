__all__ = ["AutoSwapConfig", "AutoSwapPipeline"]


def __getattr__(name: str):
	if name in __all__:
		from autoswap.pipeline import AutoSwapConfig, AutoSwapPipeline

		exports = {
			"AutoSwapConfig": AutoSwapConfig,
			"AutoSwapPipeline": AutoSwapPipeline,
		}
		return exports[name]
	raise AttributeError(name)