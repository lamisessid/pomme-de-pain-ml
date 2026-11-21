import React from 'react';
import Header from './Header';
import Sidebar from './Sidebar';

const Layout = ({ children }) => (
  <div className="min-h-screen bg-gray-50">
    <Header />
    <div className="flex">
      <Sidebar />
      <main className="flex-1 min-w-0 p-6 lg:p-8">
        <div className="max-w-7xl mx-auto">{children}</div>
      </main>
    </div>
  </div>
);

export default Layout;
