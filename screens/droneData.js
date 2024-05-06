import {View, StyleSheet, Text, Button} from 'react-native';
import React from 'react';

export default function DroneData() {
    return (
        <View style={styles.container}>
            <Text>Drone Data</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
});