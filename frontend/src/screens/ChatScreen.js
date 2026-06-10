import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StatusBar,
  Text,
  TouchableOpacity
} from 'react-native';
import { Trash2 } from 'lucide-react-native';
import ChatBubble from '../components/ChatBubble';
import MessageInput from '../components/MessageInput';
import DemographicsCard from '../components/DemographicsCard';
import { sendMessage, clearSession } from '../api/chat';

const ChatScreen = () => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      text: "👋 Hi! Welcome to WIN Home Inspection. I'm your Franchise Assistant.\nAre you ready to start your own Highly Profitable Business!\n\nChoose a prompt below to get started.",
      options: [
        "🔍 Explore WIN Franchise Opportunity",
        "💰 Costs and Investment",
        "🚀 How to Get Started"
      ],
      isUser: false
    }
  ]);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [demographics, setDemographics] = useState({});
  const flatListRef = useRef();

  const handleSend = async (text) => {
    // Add user message to UI
    const userMsg = { id: Date.now().toString(), text, isUser: true };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const data = await sendMessage(text, sessionId);

      // Update session ID if it was just created
      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
      }

      // Add agent message to UI
      const agentMsg = { id: (Date.now() + 1).toString(), text: data.answer, isUser: false };
      setMessages(prev => [...prev, agentMsg]);

      // Update demographics if returned
      if (data.demographics) {
        setDemographics(data.demographics);
      }
    } catch (error) {
      const errorMsg = { id: Date.now().toString(), text: "Sorry, I encountered an error. Please check your connection and try again.", isUser: false };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    if (sessionId) {
      await clearSession(sessionId);
    }
    setMessages([{
      id: '1',
      text: "👋 Hi! Welcome to WIN Home Inspection. I'm your Franchise Assistant.\nAre you ready to start your own Highly Profitable Business!\n\nChoose a prompt below to get started.",
      options: [
        "🔍 Explore WIN Franchise Opportunity",
        "💰 Costs and Investment",
        "🚀 How to Get Started"
      ],
      isUser: false
    }]);
    setSessionId(null);
    setDemographics({});
  };

  useEffect(() => {
    // Scroll to end when messages change
    setTimeout(() => {
      flatListRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, [messages]);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />

      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>WIN Franchise</Text>
          <Text style={styles.headerSubtitle}>Assistant</Text>
        </View>
        <TouchableOpacity onPress={handleReset} style={styles.resetButton}>
          <Trash2 color="#666" size={20} />
        </TouchableOpacity>
      </View>

      <DemographicsCard demographics={demographics} />

      <FlatList
        ref={flatListRef}
        data={messages}
        keyExtractor={item => item.id}
        renderItem={({ item }) => (
          <View>
            <ChatBubble message={item.text} isUser={item.isUser} />
            {item.options && (
              <View style={styles.optionsContainer}>
                {item.options.map((opt, idx) => (
                  <TouchableOpacity
                    key={idx}
                    style={styles.optionButton}
                    onPress={() => handleSend(opt)}
                    disabled={isLoading}
                  >
                    <Text style={styles.optionText}>{opt}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </View>
        )}
        contentContainerStyle={styles.listContent}
      />

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        <MessageInput onSend={handleSend} isLoading={isLoading} />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#0056b3',
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#666',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  resetButton: {
    padding: 8,
  },
  listContent: {
    paddingVertical: 10,
    flexGrow: 1,
  },
  optionsContainer: {
    paddingHorizontal: 16,
    paddingBottom: 8,
    alignItems: 'flex-start',
  },
  optionButton: {
    backgroundColor: '#e6f2ff',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#0056b3',
    marginVertical: 4,
    marginLeft: 8,
  },
  optionText: {
    color: '#0056b3',
    fontSize: 14,
    fontWeight: '500',
  },
});

export default ChatScreen;
