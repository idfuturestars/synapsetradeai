/**
 * SynapseTrade AI™ - Mobile App Companion
 * Chief Technical Architect Implementation
 * React Native application for iOS and Android
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  RefreshControl,
  Animated,
  Dimensions,
  StatusBar,
  Platform,
  Alert,
  Modal,
  FlatList,
  SafeAreaView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { LineChart, CandlestickChart } from 'react-native-wagmi-charts';
import Icon from 'react-native-vector-icons/MaterialIcons';
import LinearGradient from 'react-native-linear-gradient';
import PushNotification from 'react-native-push-notification';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// API Configuration
const API_BASE_URL = 'https://synapsetradeai.emergent.app/api';

// Color Theme
const colors = {
  primary: '#667eea',
  secondary: '#764ba2',
  background: '#0a0e27',
  surface: '#1a1e3a',
  text: '#ffffff',
  textSecondary: '#a0a0a0',
  success: '#4caf50',
  error: '#f44336',
  warning: '#ff9800',
};

// Configure Push Notifications
PushNotification.configure({
  onRegister: function (token) {
    console.log('TOKEN:', token);
  },
  onNotification: function (notification) {
    console.log('NOTIFICATION:', notification);
  },
  permissions: {
    alert: true,
    badge: true,
    sound: true,
  },
  popInitialNotification: true,
  requestPermissions: true,
});

// Custom Hook for API calls
const useAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = async (endpoint, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const token = await AsyncStorage.getItem('authToken');
      const headers = {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      };

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setLoading(false);
      return data;
    } catch (err) {
      setError(err.message);
      setLoading(false);
      throw err;
    }
  };

  return { request, loading, error };
};

// Animated Header Component
const AnimatedHeader = ({ title, subtitle }) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(-50)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 800,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <LinearGradient
      colors={[colors.primary, colors.secondary]}
      style={styles.header}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <Animated.View
        style={[
          styles.headerContent,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          },
        ]}
      >
        <Text style={styles.headerTitle}>{title}</Text>
        <Text style={styles.headerSubtitle}>{subtitle}</Text>
      </Animated.View>
    </LinearGradient>
  );
};

// Market Data Card
const MarketCard = ({ symbol, data, onPress }) => {
  const priceColor = data.change >= 0 ? colors.success : colors.error;
  const changeIcon = data.change >= 0 ? 'trending-up' : 'trending-down';

  return (
    <TouchableOpacity style={styles.marketCard} onPress={onPress}>
      <View style={styles.marketCardHeader}>
        <Text style={styles.marketSymbol}>{symbol}</Text>
        <Icon name={changeIcon} size={24} color={priceColor} />
      </View>
      <Text style={styles.marketPrice}>${data.price.toFixed(2)}</Text>
      <View style={styles.marketChangeContainer}>
        <Text style={[styles.marketChange, { color: priceColor }]}>
          {data.change >= 0 ? '+' : ''}{data.change.toFixed(2)}
        </Text>
        <Text style={[styles.marketChangePercent, { color: priceColor }]}>
          ({data.change_percent.toFixed(2)}%)
        </Text>
      </View>
    </TouchableOpacity>
  );
};

// Technical Indicators Component
const TechnicalIndicators = ({ data }) => {
  const indicators = [
    { label: 'RSI', value: data.rsi, threshold: [30, 70] },
    { label: 'MACD', value: data.macd.value, signal: data.macd.signal },
    { label: 'SMA 20', value: data.moving_averages.sma_20 },
    { label: 'SMA 50', value: data.moving_averages.sma_50 },
  ];

  return (
    <View style={styles.indicatorsContainer}>
      <Text style={styles.sectionTitle}>Technical Indicators</Text>
      <View style={styles.indicatorsGrid}>
        {indicators.map((indicator, index) => (
          <View key={index} style={styles.indicatorItem}>
            <Text style={styles.indicatorLabel}>{indicator.label}</Text>
            <Text style={styles.indicatorValue}>
              {typeof indicator.value === 'number'
                ? indicator.value.toFixed(2)
                : 'N/A'}
            </Text>
          </View>
        ))}
      </View>
    </View>
  );
};

// Sentiment Analysis Component
const SentimentAnalysis = ({ onAnalyze }) => {
  const [headlines, setHeadlines] = useState(['']);
  const [results, setResults] = useState(null);
  const { request, loading } = useAPI();

  const addHeadline = () => {
    setHeadlines([...headlines, '']);
  };

  const updateHeadline = (index, text) => {
    const updated = [...headlines];
    updated[index] = text;
    setHeadlines(updated);
  };

  const analyzeSentiment = async () => {
    try {
      const data = await request('/sentiment/analyze', {
        method: 'POST',
        body: JSON.stringify({ headlines: headlines.filter(h => h.trim()) }),
      });
      setResults(data);
      onAnalyze(data);
    } catch (err) {
      Alert.alert('Error', 'Failed to analyze sentiment');
    }
  };

  return (
    <View style={styles.sentimentContainer}>
      <Text style={styles.sectionTitle}>News Sentiment Analysis</Text>
      {headlines.map((headline, index) => (
        <TextInput
          key={index}
          style={styles.input}
          placeholder="Enter news headline..."
          placeholderTextColor={colors.textSecondary}
          value={headline}
          onChangeText={(text) => updateHeadline(index, text)}
        />
      ))}
      <TouchableOpacity style={styles.addButton} onPress={addHeadline}>
        <Icon name="add" size={24} color={colors.text} />
        <Text style={styles.addButtonText}>Add Headline</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.analyzeButton, loading && styles.disabledButton]}
        onPress={analyzeSentiment}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color={colors.text} />
        ) : (
          <Text style={styles.analyzeButtonText}>Analyze Sentiment</Text>
        )}
      </TouchableOpacity>
      {results && (
        <View style={styles.sentimentResults}>
          <Text style={styles.sentimentScore}>
            Average Sentiment: {results.aggregate.average_sentiment.toFixed(3)}
          </Text>
          <Text style={styles.sentimentSignal}>
            Signal: {results.aggregate.signal.toUpperCase()}
          </Text>
        </View>
      )}
    </View>
  );
};

// Portfolio Component
const Portfolio = ({ navigation }) => {
  const [portfolio, setPortfolio] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const { request } = useAPI();

  const fetchPortfolio = async () => {
    try {
      const data = await request('/portfolio');
      setPortfolio(data);
    } catch (err) {
      console.error('Portfolio fetch error:', err);
    }
  };

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchPortfolio();
    setRefreshing(false);
  };

  if (!portfolio) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={colors.primary} />
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={colors.primary}
        />
      }
    >
      <View style={styles.portfolioHeader}>
        <Text style={styles.portfolioTitle}>Total Balance</Text>
        <Text style={styles.portfolioBalance}>
          ${portfolio.portfolio.balance.toFixed(2)}
        </Text>
      </View>
      <View style={styles.tradesSection}>
        <Text style={styles.sectionTitle}>Recent Trades</Text>
        <FlatList
          data={portfolio.recent_trades}
          keyExtractor={(item) => item.id.toString()}
          renderItem={({ item }) => (
            <View style={styles.tradeItem}>
              <View style={styles.tradeHeader}>
                <Text style={styles.tradeSymbol}>{item.symbol}</Text>
                <Text style={styles.tradeAction}>{item.action.toUpperCase()}</Text>
              </View>
              <Text style={styles.tradeDetails}>
                {item.quantity} @ ${item.price.toFixed(2)}
              </Text>
              <Text style={styles.tradeTimestamp}>
                {new Date(item.timestamp).toLocaleString()}
              </Text>
            </View>
          )}
        />
      </View>
    </ScrollView>
  );
};

// Main App Component
const App = () => {
  const [activeTab, setActiveTab] = useState('markets');
  const [isConnected, setIsConnected] = useState(true);
  const [marketData, setMarketData] = useState({});
  const { request } = useAPI();

  // Network connectivity monitoring
  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setIsConnected(state.isConnected);
    });

    return () => unsubscribe();
  }, []);

  // Fetch market data
  const fetchMarketData = async () => {
    try {
      const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA'];
      const promises = symbols.map(symbol => request(`/market/${symbol}`));
      const results = await Promise.all(promises);
      
      const data = {};
      symbols.forEach((symbol, index) => {
        data[symbol] = results[index];
      });
      
      setMarketData(data);
    } catch (err) {
      console.error('Market data fetch error:', err);
    }
  };

  useEffect(() => {
    fetchMarketData();
    const interval = setInterval(fetchMarketData, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  // Setup push notifications for price alerts
  const setupPriceAlerts = (symbol, targetPrice) => {
    PushNotification.localNotificationSchedule({
      title: `Price Alert: ${symbol}`,
      message: `${symbol} has reached your target price of $${targetPrice}`,
      date: new Date(Date.now() + 60 * 1000), // 60 seconds from now
      allowWhileIdle: true,
    });
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'markets':
        return (
          <ScrollView style={styles.container}>
            <AnimatedHeader
              title="SynapseTrade AI™"
              subtitle="Advanced Trading Intelligence"
            />
            <View style={styles.marketGrid}>
              {Object.entries(marketData).map(([symbol, data]) => (
                <MarketCard
                  key={symbol}
                  symbol={symbol}
                  data={data}
                  onPress={() => {
                    // Navigate to detailed view
                  }}
                />
              ))}
            </View>
          </ScrollView>
        );
      
      case 'analysis':
        return (
          <ScrollView style={styles.container}>
            <AnimatedHeader
              title="Market Analysis"
              subtitle="AI-Powered Insights"
            />
            <SentimentAnalysis onAnalyze={(data) => console.log(data)} />
            {marketData.AAPL && (
              <TechnicalIndicators data={marketData.AAPL} />
            )}
          </ScrollView>
        );
      
      case 'portfolio':
        return <Portfolio />;
      
      default:
        return null;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />
      
      {!isConnected && (
        <View style={styles.offlineBanner}>
          <Text style={styles.offlineText}>No Internet Connection</Text>
        </View>
      )}
      
      {renderContent()}
      
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'markets' && styles.activeTab]}
          onPress={() => setActiveTab('markets')}
        >
          <Icon name="trending-up" size={24} color={colors.text} />
          <Text style={styles.tabText}>Markets</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.tab, activeTab === 'analysis' && styles.activeTab]}
          onPress={() => setActiveTab('analysis')}
        >
          <Icon name="analytics" size={24} color={colors.text} />
          <Text style={styles.tabText}>Analysis</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.tab, activeTab === 'portfolio' && styles.activeTab]}
          onPress={() => setActiveTab('portfolio')}
        >
          <Icon name="account-balance-wallet" size={24} color={colors.text} />
          <Text style={styles.tabText}>Portfolio</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    height: 200,
    justifyContent: 'center',
    alignItems: 'center',
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  headerContent: {
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: colors.text,
    marginBottom: 10,
  },
  headerSubtitle: {
    fontSize: 16,
    color: colors.text,
    opacity: 0.9,
  },
  marketGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    padding: 20,
  },
  marketCard: {
    width: (screenWidth - 60) / 2,
    backgroundColor: colors.surface,
    borderRadius: 20,
    padding: 20,
    marginBottom: 20,
  },
  marketCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  marketSymbol: {
    fontSize: 18,
    fontWeight: 'bold',
    color: colors.text,
  },
  marketPrice: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text,
    marginBottom: 5,
  },
  marketChangeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  marketChange: {
    fontSize: 16,
    fontWeight: '600',
    marginRight: 5,
  },
  marketChangePercent: {
    fontSize: 14,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.text,
    marginBottom: 15,
    paddingHorizontal: 20,
  },
  indicatorsContainer: {
    marginTop: 20,
  },
  indicatorsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 20,
  },
  indicatorItem: {
    width: (screenWidth - 60) / 2,
    backgroundColor: colors.surface,
    borderRadius: 15,
    padding: 15,
    marginBottom: 10,
    marginRight: 10,
  },
  indicatorLabel: {
    fontSize: 14,
    color: colors.textSecondary,
    marginBottom: 5,
  },
  indicatorValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: colors.text,
  },
  sentimentContainer: {
    padding: 20,
  },
  input: {
    backgroundColor: colors.surface,
    borderRadius: 10,
    padding: 15,
    color: colors.text,
    marginBottom: 10,
    fontSize: 16,
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surface,
    borderRadius: 10,
    padding: 10,
    marginBottom: 15,
  },
  addButtonText: {
    color: colors.text,
    marginLeft: 5,
    fontSize: 16,
  },
  analyzeButton: {
    backgroundColor: colors.primary,
    borderRadius: 25,
    padding: 15,
    alignItems: 'center',
  },
  analyzeButtonText: {
    color: colors.text,
    fontSize: 18,
    fontWeight: 'bold',
  },
  disabledButton: {
    opacity: 0.6,
  },
  sentimentResults: {
    marginTop: 20,
    padding: 20,
    backgroundColor: colors.surface,
    borderRadius: 15,
  },
  sentimentScore: {
    fontSize: 18,
    color: colors.text,
    marginBottom: 10,
  },
  sentimentSignal: {
    fontSize: 16,
    color: colors.primary,
    fontWeight: 'bold',
  },
  portfolioHeader: {
    padding: 20,
    alignItems: 'center',
  },
  portfolioTitle: {
    fontSize: 18,
    color: colors.textSecondary,
    marginBottom: 10,
  },
  portfolioBalance: {
    fontSize: 36,
    fontWeight: 'bold',
    color: colors.text,
  },
  tradesSection: {
    padding: 20,
  },
  tradeItem: {
    backgroundColor: colors.surface,
    borderRadius: 15,
    padding: 15,
    marginBottom: 10,
  },
  tradeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  tradeSymbol: {
    fontSize: 16,
    fontWeight: 'bold',
    color: colors.text,
  },
  tradeAction: {
    fontSize: 14,
    color: colors.primary,
    fontWeight: '600',
  },
  tradeDetails: {
    fontSize: 14,
    color: colors.textSecondary,
    marginBottom: 5,
  },
  tradeTimestamp: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.background,
    paddingBottom: Platform.OS === 'ios' ? 20 : 0,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 15,
  },
  activeTab: {
    borderTopWidth: 3,
    borderTopColor: colors.primary,
  },
  tabText: {
    fontSize: 12,
    color: colors.text,
    marginTop: 5,
  },
  offlineBanner: {
    backgroundColor: colors.warning,
    padding: 10,
    alignItems: 'center',
  },
  offlineText: {
    color: colors.text,
    fontWeight: '600',
  },
});

export default App;