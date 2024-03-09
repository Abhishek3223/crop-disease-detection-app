import React from 'react';
import { StyleSheet, Text, View, Image, Button } from 'react-native';
export default function Logs() {
    return (
        <View style={styles.container}>
            <Text style={styles.title}>Logs</Text>
            <Text style={styles.text}>This page is under construction</Text>
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
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        margin: 20
    },
    text: {
        fontSize: 16,
        margin: 20
    }
});