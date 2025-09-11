// sw.js - Service Worker for notifications
self.addEventListener('install', (event) => {
    self.skipWaiting();
    console.log('Service Worker installed');
  });
  
  self.addEventListener('activate', (event) => {
    console.log('Service Worker activated');
  });
  
  self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'NOTIFICATION') {
      self.registration.showNotification(event.data.title, {
        body: event.data.message,
        icon: 'images/notification-icon.png',
        vibrate: [200, 100, 200]
      });
    }
  });
  
  self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    event.waitUntil(
      clients.matchAll({type: 'window'}).then(windowClients => {
        for (const client of windowClients) {
          if (client.url === '/' && 'focus' in client) {
            return client.focus();
          }
        }
        if (clients.openWindow) {
          return clients.openWindow('/');
        }
      })
    );
  });