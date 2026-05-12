import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { User, Phone, MapPin, Hash } from 'lucide-react-native';

const DemographicsCard = ({ demographics }) => {
  if (!demographics || Object.keys(demographics).length === 0) return null;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Prospect Details</Text>
      <View style={styles.grid}>
        {demographics.name && (
          <View style={styles.item}>
            <User size={14} color="#666" />
            <Text style={styles.label}>{demographics.name}</Text>
          </View>
        )}
        {demographics.phone_number && (
          <View style={styles.item}>
            <Phone size={14} color="#666" />
            <Text style={styles.label}>{demographics.phone_number}</Text>
          </View>
        )}
        {demographics.pin_code && (
          <View style={styles.item}>
            <Hash size={14} color="#666" />
            <Text style={styles.label}>{demographics.pin_code}</Text>
          </View>
        )}
        {demographics.address && (
          <View style={styles.item}>
            <MapPin size={14} color="#666" />
            <Text style={styles.label}>{demographics.address}</Text>
          </View>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    margin: 12,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    backgroundColor: '#fcfcfc',
  },
  title: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#0056b3',
    textTransform: 'uppercase',
    marginBottom: 8,
    letterSpacing: 0.5,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  label: {
    fontSize: 13,
    color: '#444',
  },
});

export default DemographicsCard;
