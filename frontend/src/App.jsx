import { useState } from 'react'

const LABEL_MAP = {
  no_tumor: 'No Tumor',
  glioma: 'Glioma',
  meningioma: 'Meningioma',
  pituitary: 'Pituitary',
  'Bacterial Pneumonia': 'Bacterial Pneumonia',
  'Virus Pneumonia': 'Viral Pneumonia',
  Normal: 'Normal',
  Uncertain: 'Uncertain',
}

function prettyLabel(raw) {
  if (!raw) return raw
  return LABEL_MAP[raw] || String(raw).replaceAll('_', ' ')
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [scanType, setScanType] = useState('auto')
  const [result, setResult] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingHeatmap, setLoadingHeatmap] = useState(false)
  const [showHeatmap, setShowHeatmap] = useState(false)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
      setResult(null)
      setHeatmap(null)
      setShowHeatmap(false)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return alert('Please select an image first')
    console.log('DEBUG - scanType being sent:', scanType)

    setLoading(true)
    setResult(null)
    setHeatmap(null)
    setShowHeatmap(false)

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('scan_type', scanType)

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      alert('Error connecting to backend')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const resolveExplainType = () => {
    if (!result) return scanType
    if (result.used_model === 'chest_xray') return 'chest_xray'
    if (result.used_model === 'brain_tumor') return 'brain_tumor'
    if (result.scan_type === 'brain_tumor' || result.scan_type === 'chest_xray') {
      return result.scan_type
    }
    return scanType === 'auto' ? 'brain_tumor' : scanType
  }

  const handleShowAnalysis = async () => {
    if (!selectedFile || !result) return
    if (result.used_model === 'none' || result.prediction === 'Uncertain') {
      return alert('No model decision to explain for an Uncertain result')
    }

    setLoadingHeatmap(true)

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('scan_type', resolveExplainType())

    try {
      const response = await fetch('http://127.0.0.1:8000/explain', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setHeatmap(data.heatmap)
      setShowHeatmap(true)
    } catch (error) {
      alert('Failed to generate analysis heatmap')
      console.error(error)
    } finally {
      setLoadingHeatmap(false)
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-white text-black rounded-2xl flex items-center justify-center font-bold text-xl">
              M
            </div>
            <div>
              <h1 className="font-semibold text-2xl tracking-tight">MedLens</h1>
              <p className="text-[10px] text-zinc-500 -mt-1">AI Medical Imaging</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-5 pt-10 pb-16">
        <div className="text-center mb-10">
          <h2 className="text-5xl font-semibold tracking-tighter">Upload. Analyze. Understand.</h2>
          <p className="mt-3 text-lg text-zinc-400 max-w-md mx-auto">
            Get instant AI-powered insights from MRI and X-ray scans.
          </p>
        </div>

        <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-8">
          <div className="mb-6">
            <label className="block text-sm font-medium text-zinc-400 mb-2">Scan Type</label>
            <select
              value={scanType}
              onChange={(e) => setScanType(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-700 rounded-2xl px-4 py-3 text-white focus:outline-none focus:border-zinc-500"
            >
              <option value="auto">Automatic (Smart Mode)</option>
              <option value="brain_tumor">Brain Tumor MRI</option>
              <option value="chest_xray">Chest X-ray (Pneumonia)</option>
            </select>
          </div>

          <div
            onClick={() => document.getElementById('fileInput').click()}
            className="border border-dashed border-zinc-700 hover:border-zinc-500 transition-colors rounded-3xl p-10 text-center cursor-pointer"
          >
            <input
              id="fileInput"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />

            <div className="mx-auto w-14 h-14 bg-zinc-800 rounded-2xl flex items-center justify-center mb-4 text-3xl">
              📤
            </div>
            <p className="font-medium">Click to upload scan</p>
            <p className="text-sm text-zinc-500 mt-1">PNG or JPG • Max 10MB</p>

            {selectedFile && (
              <p className="mt-4 text-sm text-emerald-400">
                Selected: {selectedFile.name}
              </p>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className="mt-6 w-full bg-white text-black font-semibold py-3.5 rounded-2xl disabled:opacity-50 hover:bg-zinc-200 active:scale-[0.985] transition-all"
          >
            {loading ? 'Analyzing...' : 'Analyze Scan'}
          </button>
        </div>

        {result && (
          <div className="mt-8 bg-zinc-900 border border-zinc-800 rounded-3xl p-8">
            <h3 className="font-semibold text-xl mb-4">Analysis Result</h3>

            <div className="space-y-4">
              <div>
                <p className="text-sm text-zinc-400">Prediction</p>
                <p className="text-3xl font-semibold tracking-tight">
                  {prettyLabel(result.prediction)}
                </p>
              </div>

              <div>
                <p className="text-sm text-zinc-400">Confidence</p>
                <p className="text-4xl font-semibold text-emerald-400">{result.confidence}%</p>
              </div>

              {result.used_model && result.used_model !== 'none' && (
                <div className="text-sm">
                  <span className="text-zinc-400">Model used: </span>
                  <span className="font-medium text-violet-400">
                    {String(result.used_model).includes('chest')
                      ? 'Chest X-ray Model'
                      : 'Brain Tumor Model'}
                  </span>
                </div>
              )}

              {result.warning && (
                <div className="bg-yellow-950 border border-yellow-900 text-yellow-400 px-4 py-3 rounded-2xl text-sm flex items-start gap-2">
                  <span>⚠️</span>
                  <span>{result.warning}</span>
                </div>
              )}
            </div>

            {!showHeatmap && (
              <button
                onClick={handleShowAnalysis}
                disabled={loadingHeatmap}
                className="mt-6 w-full border border-zinc-700 hover:bg-zinc-800 py-3 rounded-2xl text-sm font-medium transition-colors"
              >
                {loadingHeatmap ? 'Generating analysis...' : 'Click to show analysis'}
              </button>
            )}

            {showHeatmap && heatmap && (
              <div className="mt-6">
                <p className="text-sm text-zinc-400 mb-2">Model Focus Area (Grad-CAM)</p>
                <div className="rounded-2xl overflow-hidden border border-zinc-700">
                  <img src={heatmap} alt="Grad-CAM Heatmap" className="w-full" />
                </div>
                <p className="text-xs text-zinc-500 mt-2 text-center">
                  Brighter areas = Stronger influence on the prediction
                </p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
