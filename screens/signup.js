import { Text, View, Image, Button, StyleSheet, TextInput, Press } from 'react-native';
import React from 'react';
import { useNavigation } from '@react-navigation/native';
import {app} from '../firebase.config';
import { getAuth, createUserWithEmailAndPassword } from "firebase/auth";

const auth = getAuth(app);

export default function Signup() {

    const [username, setUsername] = React.useState('');
    const [password, setPassword] = React.useState('');
    const [isLogin, setLogin] = React.useState(false);

    const navigation = useNavigation();
    return (
        <View style={styles.container}>
            <Text style={styles.header}>{isLogin ? "Login" : "Signup"}</Text>
            <View>
                <TextInput
                    style={styles.input}
                    value={username}
                    onChangeText={onChangeText = (text) => { setUsername(text) }}
                    placeholder="Username"
                    keyboardType="default"
                />
                <TextInput
                    style={styles.input}
                    onChangeText={onChangeText = (text) => { setPassword(text) }}
                    placeholder="Password"
                    keyboardType="default"
                    secureTextEntry
                />
                <Button
                    title={isLogin ? "Login" : "Signup"}
                    color="#50c878"
                    onPress={() => { handleClick() }}
                />
                {isLogin? null:<Text style={styles.loginText}>Already have an account?  <Text style={styles.login} onPress={() => { setLogin(true) }}>Login</Text></Text>}
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
    },
            login: {
                color: '#50c878'
    },
            loginText: {
                paddingTop: 20,
            fontSize: 15,
            alignSelf: 'center'
    }
});