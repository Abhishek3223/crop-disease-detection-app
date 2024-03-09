import { Text, View, Image, Button, StyleSheet, TextInput, Press } from 'react-native';
import React from 'react';
import { useNavigation } from '@react-navigation/native';
import auth from '@react-native-firebase/auth';
import firestore from '@react-native-firebase/firestore';
import { useAuth } from './context';


export default function Signup() {

    // const [username, setUsername] = React.useState('');
    const [number, setNumber] = React.useState('');
    const [code, setCode] = React.useState('')
    const [isLogin, setLogin] = React.useState(false);
    const [confirm, setConfirm] = React.useState(null);
    const navigation = useNavigation();
    const {setUser} = useAuth();

    async function signInWithPhoneNumber() {
        try {
            const confirmation = await auth().signInWithPhoneNumber(number);
            setConfirm(confirmation);
        }
        catch (e) {
            console.log("Error sending code: ", e);
        }
    };

    const confirmCode = async () => {
        try {
            const userCredential = await confirm.confirm(code);
            const user = userCredential.user;
            //check if user is new or existing
            const userDocument = await firestore().collection('users').doc(user.uid).get();
            if (userDocument.exists) {
                setUser(userDocument.data());
                navigation.reset({
                    index: 0,
                    routes: [{ name: 'HomePage' }],
                })
            }
            else {
                navigation.reset({
                    index: 0,
                    routes: [{ name: 'User_Info', params: { uid: user.uid }}],
                })
            }
        }
        catch (e) {
            console.log("Error confirming code: ", e);
        }
    };

    return (
        <View style={styles.container}>
            <View>
                {/* <TextInput
                    style={styles.input}
                    value={username}
                    onChangeText={onChangeText = (text) => { setUsername(text) }}
                    placeholder="Username"
                    keyboardType="default"
                /> */}
                {!confirm ?
                    (
                        <>
                            <Text style={styles.header}>Welcome to Planti ❤️</Text>
                            <TextInput
                                style={styles.input}
                                value={number}
                                onChangeText={onChangeText = (text) => { setNumber(text) }}
                                placeholder="Phone Number"
                                keyboardType="phone-pad"
                            />
                            <Button
                                title="Send Code"
                                color="#50c878"
                                onPress={() => { signInWithPhoneNumber() }}
                            />
                        </>
                    )
                    :
                    (<>
                        <Text style={styles.header}>
                            Enter the code sent to your number
                        </Text>
                        <TextInput
                            style={styles.input}
                            value={code}
                            onChangeText={onChangeText = (text) => { setCode(text) }}
                            placeholder="Otp"
                            keyboardType="number-pad"
                            secureTextEntry
                        />
                        <Button
                            title="Submit"
                            color="#50c878"
                            onPress={() => { confirmCode()}}
                        />
                    </>)
                }
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
        margin: 20,
        borderWidth: 1,
        borderRadius: 100,
        borderColor: '#50c878',
        padding: 10,
        width: 200
    },
    header: {
        paddingBottom:20,
        fontSize: 18,
        fontWeight: 'medium',
        alignSelf: 'center'
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