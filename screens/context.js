// context/AuthContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';
import auth from '@react-native-firebase/auth';

const AuthContext = createContext(<undefined/>);

export const useAuth = () => {
  return useContext(AuthContext);
};


export const AuthProvider = ({ children }) => {

  const [user, setUser] = useState(null);
//   useEffect(() => {
//     const unsubscribe = auth().onAuthStateChanged(currentUser => {
//       if (currentUser) {
//         // If user is already authenticated, set the user information
//         // You can customize this logic according to your application's requirements
//         setUser(currentUser);
//       }
//     });

//     return unsubscribe;
//   }, []);

  return (
    <AuthContext.Provider value={{ user, setUser }}>
      {children}
    </AuthContext.Provider>
  );
};
