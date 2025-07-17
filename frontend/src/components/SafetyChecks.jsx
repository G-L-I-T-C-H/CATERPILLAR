import React, { useState } from 'react';
import { 
  ArrowLeft, 
  Shield, 
  Eye, 
  EyeOff, 
  Mic, 
  MicOff,
  RotateCcw,
  CheckCircle
} from 'lucide-react';

const SafetyChecks = ({ onBack }) => {
  const [voiceAlertsEnabled, setVoiceAlertsEnabled] = useState(false);
  const [isRunningCheck, setIsRunningCheck] = useState(false);

  const handleRunCheck = () => {
    setIsRunningCheck(true);
    // Simulate check process
    setTimeout(() => {
      setIsRunningCheck(false);
    }, 2000);
  };

  const toggleVoiceAlerts = () => {
    setVoiceAlertsEnabled(!voiceAlertsEnabled);
  };

  return (
    <div className="min-h-screen bg-gray-800 p-4">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button 
            onClick={onBack}
            className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Dashboard</span>
          </button>
          
          <h1 className="text-3xl font-bold text-white">Safety Checks</h1>
          
          <div className="flex items-center gap-4">
            <button 
              onClick={toggleVoiceAlerts}
              className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-white transition-colors"
            >
              {voiceAlertsEnabled ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
              <span>Voice Alerts {voiceAlertsEnabled ? 'On' : 'Off'}</span>
            </button>
            
            <button 
              onClick={handleRunCheck}
              disabled={isRunningCheck}
              className="flex items-center gap-2 bg-orange-500 hover:bg-orange-600 disabled:bg-orange-400 px-6 py-2 rounded-lg text-white font-semibold transition-colors"
            >
              {isRunningCheck ? (
                <>
                  <RotateCcw className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <RotateCcw className="w-4 h-4" />
                  Run Check
                </>
              )}
            </button>
          </div>
        </div>

        {/* Last Safety Check */}
        <div className="bg-gray-700 rounded-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-white mb-2">Last Safety Check</h2>
          <p className="text-gray-400">Performed at 5:07:37 PM</p>
        </div>

        {/* Safety Check Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          
          {/* Seatbelt Check */}
          <div className="bg-gray-700 rounded-lg p-6 border-2 border-green-500">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Shield className="w-6 h-6 text-white" />
                <h3 className="text-xl font-semibold text-white">Seatbelt Check</h3>
              </div>
              <span className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                OK
              </span>
            </div>
            <p className="text-gray-300">Seatbelt properly fastened</p>
          </div>

          {/* Fatigue Detection */}
          <div className="bg-gray-700 rounded-lg p-6 border-2 border-green-500">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Eye className="w-6 h-6 text-white" />
                <h3 className="text-xl font-semibold text-white">Fatigue Detection</h3>
              </div>
              <span className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                OK
              </span>
            </div>
            <div className="mb-3">
              <p className="text-gray-300 mb-2">Fatigue level: 15%</p>
              <div className="w-full bg-gray-600 rounded-full h-2">
                <div className="bg-orange-400 h-2 rounded-full" style={{ width: '15%' }}></div>
              </div>
            </div>
          </div>

          {/* Visibility Check */}
          <div className="bg-gray-700 rounded-lg p-6 border-2 border-green-500">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Eye className="w-6 h-6 text-white" />
                <h3 className="text-xl font-semibold text-white">Visibility Check</h3>
              </div>
              <span className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                OK
              </span>
            </div>
            <p className="text-gray-300">All mirrors and windows clear</p>
          </div>
        </div>

        {/* Safety Protocols */}
        <div className="bg-gray-700 rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-orange-400 mb-6">Safety Protocols</h2>
          
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-300">Always perform safety checks before starting operations</p>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-300">Take breaks every 2 hours to prevent fatigue</p>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-300">Report any safety concerns immediately</p>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-300">Keep emergency contacts readily available</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SafetyChecks;