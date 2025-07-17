import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

import LoginScreen from './components/LoginScreen'
import Dashboard from './components/Dashboard'
import TaskScheduling from './components/TaskScheduling'
import SafetyChecks from './components/SafetyChecks'
import TrainingProgress from './components/TrainingProgress'

function App() {
  const [currentPage, setCurrentPage] = useState('login') // 'login', 'dashboard', 'tasks', 'safety'

  const handleLogin = () => {
    setCurrentPage('dashboard')
  }

  const handleNavigateToTasks = () => {
    setCurrentPage('tasks')
  }

  const handleNavigateToSafety = () => {
    setCurrentPage('safety')
  }

  const handleNavigateToTraining = () => {
    setCurrentPage('training')
  }

  const handleBackToDashboard = () => {
    setCurrentPage('dashboard')
  }

  const handleLogout = () => {
    setCurrentPage('login')
  }

  return (
    <div className="App">
      {currentPage === 'login' && (
        <LoginScreen onLogin={handleLogin} />
      )}
      {currentPage === 'dashboard' && (
        <Dashboard 
          onNavigateToTasks={handleNavigateToTasks}
          onNavigateToSafety={handleNavigateToSafety}
          onNavigateToTraining={handleNavigateToTraining}
          onLogout={handleLogout}
        />
      )}
      {currentPage === 'tasks' && (
        <TaskScheduling onBack={handleBackToDashboard} />
      )}
      {currentPage === 'safety' && (
        <SafetyChecks onBack={handleBackToDashboard} />
      )}
      {currentPage === 'training' && (
        <TrainingProgress onBack={handleBackToDashboard} />
      )}
    </div>
  )
}

export default App