import axios from 'axios';

// For local testing on iOS simulator, localhost works.
// For physical devices, replace with your computer's local IP (e.g., 192.168.1.XX)
// const BASE_URL = 'http://localhost:9050';

// Determine the backend URL dynamically from the current browser environment.
// If loaded on an EC2 instance at http://<ec2-ip>:9100, it will hit http://<ec2-ip>:9050.
const getBaseUrl = () => {
  if (typeof window !== 'undefined' && window.location && window.location.hostname) {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    return `${protocol}//${hostname}:9050`;
  }
  // Fallback to localhost for native mobile environments (iOS/Android simulators)
  return 'http://localhost:9050';
};

const BASE_URL = getBaseUrl();

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendMessage = async (query, sessionId = null) => {
  try {
    const response = await api.post('/chat', {
      query,
      session_id: sessionId,
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const clearSession = async (sessionId) => {
  try {
    const response = await api.delete(`/chat/${sessionId}`);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};
