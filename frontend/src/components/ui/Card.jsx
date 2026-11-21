import React from 'react';
import { clsx } from 'clsx';

const Card = ({ 
  children, 
  className = '',
  padding = true,
  hover = false,
  ...props 
}) => {
  return (
    <div
      className={clsx(
        'bg-white rounded-xl shadow-sm border border-gray-200',
        padding && 'p-6',
        hover && 'hover:shadow-md transition-shadow',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

const CardHeader = ({ children, className = '', ...props }) => (
  <div className={clsx('mb-6', className)} {...props}>
    {children}
  </div>
);

const CardTitle = ({ children, className = '', ...props }) => (
  <h3 className={clsx('text-lg font-semibold text-gray-900', className)} {...props}>
    {children}
  </h3>
);

const CardDescription = ({ children, className = '', ...props }) => (
  <p className={clsx('text-sm text-gray-600 mt-1', className)} {...props}>
    {children}
  </p>
);

const CardContent = ({ children, className = '', ...props }) => (
  <div className={clsx(className)} {...props}>
    {children}
  </div>
);

const CardFooter = ({ children, className = '', ...props }) => (
  <div className={clsx('mt-6 pt-6 border-t border-gray-200', className)} {...props}>
    {children}
  </div>
);

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter };
