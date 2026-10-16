import React from 'react';

interface ProbabilityBarChartProps {
  probabilities: { className: string; probability: number; percentage: number }[];
  predictedClass: string;
  thresholdApplied?: number;
}

export const ProbabilityBarChart: React.FC<ProbabilityBarChartProps> = ({
  probabilities,
  predictedClass,
  thresholdApplied,
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
          <span>Class Probabilities (%)</span>
          {thresholdApplied !== undefined && (
            <span className="text-xs font-normal text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
              Calibrated cutoff: {(thresholdApplied * 100).toFixed(0)}%
            </span>
          )}
        </h4>
        <span className="text-xs text-slate-400">Softmax Normalized</span>
      </div>

      <div className="space-y-3">
        {probabilities.map((item) => {
          const isWinner = item.className === predictedClass;
          return (
            <div key={item.className} className="space-y-1">
              <div className="flex justify-between items-center text-xs">
                <span
                  className={`font-medium ${
                    isWinner ? 'text-blue-700 font-semibold' : 'text-slate-600'
                  }`}
                >
                  {item.className}
                  {isWinner && (
                    <span className="ml-1.5 px-1.5 py-0.2 rounded text-[10px] bg-blue-100 text-blue-800 font-bold">
                      TOP
                    </span>
                  )}
                </span>
                <span
                  className={`font-mono ${
                    isWinner ? 'text-blue-800 font-bold' : 'text-slate-500'
                  }`}
                >
                  {item.percentage.toFixed(1)}%
                </span>
              </div>

              <div className="relative w-full h-4 bg-slate-100 rounded-full overflow-hidden">
                {/* Optional threshold marker for binary */}
                {thresholdApplied !== undefined && item.className === 'PNEUMONIA' && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-rose-500 z-10"
                    style={{ left: `${thresholdApplied * 100}%` }}
                    title={`Decision threshold: ${(thresholdApplied * 100).toFixed(0)}%`}
                  />
                )}
                <div
                  className={`h-full rounded-full transition-all duration-500 ease-out ${
                    isWinner ? 'bg-blue-600' : 'bg-slate-300'
                  }`}
                  style={{ width: `${Math.max(item.percentage, 1)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
        <span>0%</span>
        <span>25%</span>
        <span>50%</span>
        <span>75%</span>
        <span>100%</span>
      </div>
    </div>
  );
};
