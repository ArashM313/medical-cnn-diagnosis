import React from 'react';

interface ConfusionMatrixProps {
  classes: string[];
  matrix: number[][];
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ classes, matrix }) => {
  // Find maximum cell value for normalizing Blues color intensity
  let maxVal = 1;
  let totalSamples = 0;
  let correctSamples = 0;

  for (let r = 0; r < matrix.length; r++) {
    for (let c = 0; c < matrix[r].length; c++) {
      const val = matrix[r][c];
      if (val > maxVal) maxVal = val;
      totalSamples += val;
      if (r === c) correctSamples += val;
    }
  }

  const overallAccuracy = totalSamples > 0 ? (correctSamples / totalSamples) * 100 : 0;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-slate-800">
            Confusion Matrix (Test Set)
          </h4>
          <p className="text-xs text-slate-500 mt-0.5">
            Rows: Actual Ground Truth &nbsp;|&nbsp; Columns: Model Predicted
          </p>
        </div>
        <div className="text-right">
          <span className="text-xs text-slate-500">Test Accuracy</span>
          <div className="text-sm font-bold text-blue-700 font-mono">
            {overallAccuracy.toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full align-middle">
          {/* Column labels (Predicted) */}
          <div className="text-center text-xs font-semibold text-slate-600 mb-1 tracking-wider uppercase">
            Predicted Class
          </div>

          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 text-xs font-medium text-slate-400 text-right pr-3">
                  Actual \ Pred
                </th>
                {classes.map((cls) => (
                  <th
                    key={cls}
                    className="p-2 text-xs font-medium text-slate-700 text-center truncate max-w-[100px]"
                    title={cls}
                  >
                    {cls}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rIdx) => {
                const actualClass = classes[rIdx];
                const rowTotal = row.reduce((a, b) => a + b, 0);

                return (
                  <tr key={actualClass}>
                    <th className="p-2 text-xs font-semibold text-slate-700 text-right pr-3 whitespace-nowrap bg-slate-50/50 rounded-l">
                      {actualClass}
                    </th>
                    {row.map((cellVal, cIdx) => {
                      const isDiagonal = rIdx === cIdx;
                      const intensity = cellVal / maxVal;
                      // Generate background color on the "Blues" palette
                      // Intensity range from light tint to deep royal blue
                      const bgAlpha = Math.max(0.06, intensity);
                      const textColor = intensity > 0.55 ? 'text-white' : 'text-slate-900';
                      const cellPct = rowTotal > 0 ? ((cellVal / rowTotal) * 100).toFixed(0) : '0';

                      return (
                        <td
                          key={cIdx}
                          className="p-3 text-center border border-slate-100 transition-all hover:ring-2 hover:ring-blue-400 group relative cursor-default"
                          style={{
                            backgroundColor: `rgba(29, 78, 216, ${bgAlpha})`,
                          }}
                          title={`Actual: ${actualClass}, Predicted: ${classes[cIdx]} -> ${cellVal} cases (${cellPct}%)`}
                        >
                          <div className={`text-sm font-mono font-bold ${textColor}`}>
                            {cellVal}
                          </div>
                          <div
                            className={`text-[10px] font-mono opacity-80 ${textColor}`}
                          >
                            {cellPct}%
                          </div>
                          {isDiagonal && (
                            <div className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-emerald-400 opacity-80" />
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
          <span>Diagonal elements indicate correct predictions</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] text-slate-400">Total samples: {totalSamples}</span>
        </div>
      </div>
    </div>
  );
};
