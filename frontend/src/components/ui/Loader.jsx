import React from 'react';

const Loader = ({ size = 'medium', className = '' }) => {
  const sizes = {
    small: 'h-4 w-4',
    medium: 'h-8 w-8',
    large: 'h-12 w-12',
  };

  return (
    <div className={`animate-spin rounded-full border-b-2 border-primary-600 ${sizes[size]} ${className}`}></div>
  );
};

export default Loader;

export const PageLoader = () => (
  <div className="min-h-[400px] flex items-center justify-center">
    <div className="text-center">
      <Loader size="large" />
      <p className="mt-4 text-gray-600">Chargement...</p>
    </div>
  </div>
);
