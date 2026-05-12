import axios from 'axios';

// For local testing on iOS simulator, localhost works.
// For physical devices, replace with your computer's local IP (e.g., 192.168.1.XX)
const BASE_URL = 'http://localhost:8000';

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
