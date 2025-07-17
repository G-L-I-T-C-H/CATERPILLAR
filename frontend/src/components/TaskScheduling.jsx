import React from 'react';
import { 
  ArrowLeft, 
  Calendar, 
  Clock, 
  MapPin, 
  CheckCircle 
} from 'lucide-react';

const TaskScheduling = ({ onBack }) => {
  const currentDate = new Date().toLocaleDateString('en-US', {
    month: 'numeric',
    day: 'numeric',
    year: 'numeric'
  });

  const handleMarkComplete = (taskId) => {
    console.log(`Marking task ${taskId} as complete`);
    // Add your completion logic here
  };

  return (
    <div className="min-h-screen bg-gray-800 p-4">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <button 
              onClick={onBack}
              className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Dashboard
            </button>
          </div>
          <div className="text-gray-400">
            Today: {currentDate}
          </div>
        </div>

        <h1 className="text-3xl font-bold text-white mb-8">Task Scheduling</h1>

        {/* Today's Progress */}
        <div className="bg-gray-700 rounded-lg p-6 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Calendar className="w-5 h-5 text-white" />
            <h2 className="text-xl font-semibold text-white">Today's Progress</h2>
          </div>
          
          <p className="text-gray-300 mb-4">1 of 4 tasks completed</p>
          
          {/* Progress Bar */}
          <div className="w-full bg-gray-600 rounded-full h-3 mb-2">
            <div className="bg-orange-500 h-3 rounded-full" style={{ width: '25%' }}></div>
          </div>
          
          <p className="text-gray-400 text-sm">25% Complete</p>
        </div>

        {/* Info Cards Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          
          {/* Shift Details */}
          <div className="bg-gray-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Shift Details</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-300">Start Time:</span>
                <span className="text-white font-medium">07:00 AM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">End Time:</span>
                <span className="text-white font-medium">06:00 PM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Break Time:</span>
                <span className="text-white font-medium">12:00 - 1:00 PM</span>
              </div>
            </div>
          </div>

          {/* Current Status */}
          <div className="bg-gray-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Current Status</h3>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-white">On shift for 4h 30m</span>
            </div>
            <span className="bg-green-600 text-green-100 px-3 py-1 rounded-full text-sm font-medium">
              Active
            </span>
          </div>

          {/* Next Break */}
          <div className="bg-gray-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Next Break</h3>
            <div className="text-3xl font-bold text-orange-400 mb-2">2h 15m</div>
            <p className="text-gray-300">Until lunch break</p>
          </div>
        </div>

        {/* Today's Tasks */}
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-white mb-6">Today's Tasks</h2>
          
          <div className="space-y-4">
            
            {/* Task 1 - Completed */}
            <div className="bg-gray-700 rounded-lg p-6 border-l-4 border-green-500">
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">Load Materials</h3>
                  <p className="text-gray-300 mb-4">Load construction materials from storage area to site A</p>
                  
                  <div className="flex items-center gap-6 text-sm text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      08:00 - 09:30
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-4 h-4" />
                      Storage Area B
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      Due: 10:00:00 AM
                    </div>
                  </div>
                </div>
                
                <span className="bg-green-600 text-green-100 px-3 py-1 rounded-full text-sm font-medium">
                  COMPLETED
                </span>
              </div>
            </div>

            {/* Task 2 - In Progress */}
            <div className="bg-gray-700 rounded-lg p-6 border-l-4 border-blue-500">
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">Excavation Work</h3>
                  <p className="text-gray-300 mb-4">Excavate foundation area for new building section</p>
                  
                  <div className="flex items-center gap-6 text-sm text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      10:00 - 12:00
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-4 h-4" />
                      Site A - North Section
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      Due: 1:00:00 PM
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <span className="bg-blue-600 text-blue-100 px-3 py-1 rounded-full text-sm font-medium">
                    IN PROGRESS
                  </span>
                  <button
                    onClick={() => handleMarkComplete('excavation')}
                    className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Mark Complete
                  </button>
                </div>
              </div>
            </div>

            {/* Task 3 - Pending */}
            <div className="bg-gray-700 rounded-lg p-6 border-l-4 border-orange-500">
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">Material Transport</h3>
                  <p className="text-gray-300 mb-4">Transport gravel from quarry to construction site</p>
                  
                  <div className="flex items-center gap-6 text-sm text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      14:30 - 16:00
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-4 h-4" />
                      Quarry Site
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      Due: 5:00:00 PM
                    </div>
                  </div>
                </div>
                
                <span className="bg-orange-600 text-orange-100 px-3 py-1 rounded-full text-sm font-medium">
                  PENDING
                </span>
              </div>
            </div>

            {/* Task 4 - Pending */}
            <div className="bg-gray-700 rounded-lg p-6 border-l-4 border-orange-500">
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">Equipment Maintenance</h3>
                  <p className="text-gray-300 mb-4">Routine maintenance check and oil change</p>
                  
                  <div className="flex items-center gap-6 text-sm text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      16:30 - 17:30
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-4 h-4" />
                      Maintenance Bay
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      Due: 6:00:00 PM
                    </div>
                  </div>
                </div>
                
                <span className="bg-orange-600 text-orange-100 px-3 py-1 rounded-full text-sm font-medium">
                  PENDING
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TaskScheduling;