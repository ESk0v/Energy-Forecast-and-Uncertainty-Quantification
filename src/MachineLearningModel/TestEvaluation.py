from EvaluateModel import (
    PredictionsVsActualsPlot,
    ResidualsPlot,
    FirstWeekPredictionPlot,
    FirstWeekPredictionMCPlot
)

def main():
    """Run evaluation and generate plots for the trained LSTM model."""

    print("\nGenerating plots...")
    ResidualsPlot(show_plots=True)
    PredictionsVsActualsPlot(show_plots=True)
    # We only have the training and validation losses when training the model, so we don't test TrainingValidationPlot()
    FirstWeekPredictionPlot(show_plots=True)
    FirstWeekPredictionMCPlot(show_plots=True)

if __name__ == '__main__':
    main()