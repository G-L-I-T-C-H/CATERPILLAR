import React, { useState } from 'react';
import { Shield } from 'lucide-react';

const LoginScreen = ({ onLogin }) => {
  const [driverId, setDriverId] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = () => {
    // Basic validation - you can add more sophisticated logic here
    if (driverId.trim() && password.trim()) {
      console.log('Login successful with:', { driverId, password });
      // Call the onLogin function to navigate to dashboard
      onLogin();
    } else {
      alert('Please enter both Driver ID and Password');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-gray-700 rounded-lg shadow-xl p-8 w-full max-w-md">
        {/* Header with Shield Icon */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-orange-500 rounded-full mb-4">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-200 mb-2">
            Construction Driver Portal
          </h1>
          <p className="text-gray-400">
            Secure access to your driver assistance system
          </p>
        </div>

        {/* Login Form */}
        <div className="space-y-6">
          <div>
            <label htmlFor="driverId" className="block text-sm font-medium text-gray-200 mb-2">
              Driver ID
            </label>
            <input
              type="text"
              id="driverId"
              value={driverId}
              onChange={(e) => setDriverId(e.target.value)}
              placeholder="Enter your driver ID"
              className="w-full px-4 py-3 bg-gray-600 text-white placeholder-gray-400 rounded-lg border border-gray-500 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-200 mb-2">
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              className="w-full px-4 py-3 bg-gray-600 text-white placeholder-gray-400 rounded-lg border border-gray-500 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
            />
          </div>

          <button
            onClick={handleSubmit}
            className="w-full bg-orange-500 hover:bg-orange-600 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 ease-in-out transform hover:scale-105"
          >
            Sign In
          </button>
        </div>

        {/* Demo Notice */}
        <div className="mt-6 bg-yellow-100 border border-yellow-400 text-yellow-800 px-4 py-3 rounded-lg">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-2">
              <span className="font-medium">Demo:</span> Use any Driver ID and password to login
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginScreen;