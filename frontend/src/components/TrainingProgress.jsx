import React, { useState } from 'react';
import { 
  GraduationCap, 
  Clock, 
  CheckCircle, 
  Play,
  ArrowLeft,
  ArrowRight,
  Book,
  Users,
  Settings,
  Wrench
} from 'lucide-react';

const TrainingProgress = ({ onBack }) => {
  const [activeCategory, setActiveCategory] = useState('All Categories');

  const categories = ['All Categories', 'Safety', 'Operations', 'Maintenance', 'Technical'];

  const trainingModules = [
    {
      id: 1,
      title: "Comprehensive guide to construction equipment safety",
      level: "beginner",
      duration: "15 min",
      completed: true,
      icon: <Book className="w-5 h-5" />
    },
    {
      id: 2,
      title: "Safety inspection procedures before equipment operation",
      level: "beginner", 
      duration: "8 min",
      completed: true,
      icon: <Book className="w-5 h-5" />
    },
    {
      id: 3,
      title: "Advanced Hydraulic Systems",
      subtitle: "Understanding and troubleshooting hydraulic systems",
      level: "advanced",
      duration: "25 min",
      completed: false,
      icon: <Book className="w-5 h-5" />
    },
    {
      id: 4,
      title: "Load Management and Weight Distribution",
      subtitle: "Proper techniques for load handling and balance",
      level: "intermediate",
      duration: "20 min",
      completed: false,
      icon: <Book className="w-5 h-5" />
    },
    {
      id: 5,
      title: "Emergency Procedures Manual",
      subtitle: "Step-by-step emergency response procedures",
      level: "beginner",
      duration: "12 min",
      completed: false,
      icon: <Book className="w-5 h-5" />
    },
    {
      id: 6,
      title: "Equipment Maintenance Audio Guide",
      subtitle: "Audio guide for routine maintenance procedures",
      level: "intermediate",
      duration: "18 min",
      completed: false,
      icon: <Book className="w-5 h-5" />
    }
  ];

  const learningPath = [
    { id: 1, title: "Basic Safety Protocols", completed: true },
    { id: 2, title: "Equipment Operation Fundamentals", completed: true },
    { id: 3, title: "Load Management Techniques", status: "next" },
    { id: 4, title: "Advanced Hydraulic Systems", status: "upcoming" }
  ];

  const getLevelColor = (level) => {
    switch(level) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getIconForCategory = (category) => {
    switch(category) {
      case 'Safety': return <Settings className="w-4 h-4" />;
      case 'Operations': return <Play className="w-4 h-4" />;
      case 'Maintenance': return <Wrench className="w-4 h-4" />;
      case 'Technical': return <Book className="w-4 h-4" />;
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-800 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button 
              onClick={onBack}
              className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-white" />
            </button>
            <h1 className="text-3xl font-bold text-white">Training & Tutorials</h1>
          </div>
          <div className="bg-blue-600 text-white px-4 py-2 rounded-full flex items-center gap-2">
            <GraduationCap className="w-4 h-4" />
            <span>Certified Driver</span>
          </div>
        </div>

        {/* Training Progress Card */}
        <div className="bg-gray-100 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Book className="w-5 h-5 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-800">Training Progress</h2>
          </div>
          <p className="text-gray-600 mb-4">2 of 6 modules completed</p>
          
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Progress</span>
              <span className="text-sm font-semibold text-orange-600">33% Complete</span>
            </div>
            <div className="w-full bg-gray-300 rounded-full h-2">
              <div className="bg-orange-500 h-2 rounded-full" style={{ width: '33%' }}></div>
            </div>
          </div>
        </div>

        {/* Category Filters */}
        <div className="flex gap-2 flex-wrap">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setActiveCategory(category)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                activeCategory === category
                  ? 'bg-orange-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {getIconForCategory(category)}
              {category}
            </button>
          ))}
        </div>

        {/* Training Modules Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {trainingModules.map((module) => (
            <div key={module.id} className="bg-gray-700 rounded-lg p-6">
              <div className="flex items-start justify-between mb-4">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${getLevelColor(module.level)}`}>
                  {module.level}
                </span>
                {module.completed && (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                )}
              </div>

              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  {module.icon}
                  <span className="text-gray-400 text-sm">Course</span>
                </div>
                <h3 className="text-white font-semibold mb-2">{module.title}</h3>
                {module.subtitle && (
                  <p className="text-gray-300 text-sm mb-2">{module.subtitle}</p>
                )}
                <div className="flex items-center gap-2 text-gray-400 text-sm">
                  <Clock className="w-4 h-4" />
                  <span>{module.duration}</span>
                </div>
              </div>

              <button 
                className={`w-full py-3 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                  module.completed
                    ? 'bg-green-600 text-green-100'
                    : 'bg-orange-500 hover:bg-orange-600 text-white'
                }`}
              >
                {module.completed ? (
                  <>
                    <CheckCircle className="w-5 h-5" />
                    Completed
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start Training
                  </>
                )}
              </button>
            </div>
          ))}
        </div>

        {/* Recommended Learning Path */}
        <div className="bg-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-blue-400 mb-2">Recommended Learning Path</h2>
          <p className="text-gray-300 mb-6">Complete these modules in order for optimal learning</p>
          
          <div className="space-y-4">
            {learningPath.map((item, index) => (
              <div key={item.id} className="flex items-center justify-between p-4 bg-gray-600 rounded-lg">
                <div className="flex items-center gap-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    item.completed 
                      ? 'bg-green-600 text-white' 
                      : item.status === 'next'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-500 text-gray-300'
                  }`}>
                    {item.completed ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : (
                      <span className="text-sm font-bold">{index + 1}</span>
                    )}
                  </div>
                  <span className="text-white font-medium">{item.title}</span>
                </div>
                
                <div className="flex items-center gap-2">
                  {item.completed && (
                    <span className="bg-green-600 text-green-100 px-3 py-1 rounded-full text-xs font-medium">
                      Completed
                    </span>
                  )}
                  {item.status === 'next' && (
                    <span className="bg-blue-600 text-blue-100 px-3 py-1 rounded-full text-xs font-medium">
                      Next
                    </span>
                  )}
                  {item.status === 'upcoming' && (
                    <span className="bg-gray-500 text-gray-100 px-3 py-1 rounded-full text-xs font-medium">
                      Upcoming
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingProgress;