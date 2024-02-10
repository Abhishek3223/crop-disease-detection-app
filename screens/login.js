import {Text, View, Image, Button, StyleSheet, TextInput} from 'react-native';
import React from 'react';
import {useNavigation} from '@react-navigation/native';

export default function Login() {

    const navigation = useNavigation();
    const [username, setUsername] = React.useState('');
    const [password, setPassword] = React.useState('');

    function handleClick() {
        console.log(username);
    }

    return (
        <View style={styles.container}>
            <Text style={styles.header}>Login</Text>
            <View>
                <TextInput
                    style={styles.input}
                    onChangeText={onChangeText = (text) => { setUsername(text) }}
                    placeholder="Username"
                    keyboardType="default"
                />
                <TextInput
                    style={styles.input}
                    onChangeText={onChangeText = (text) => { setPassword(text) }}
                    placeholder="Password"
                    keyboardType="default"
                />
                <Button
                    title="Login"
                    color="#50c878"
                    onPress={() => { handleClick() }}
                />
            </View>
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
    input: {
        height: 50,
        margin: 12,
        borderWidth: 1,
        borderRadius: 100,
        borderColor: '#50c878',
        padding: 10,
        width: 200
    },
    header: {
        paddingVertical: 25,
        fontSize: 18,
        fontWeight: 'medium',
        alignSelf: 'center'
    }
});