interface Config {
    apiUrl: string;
}

const devConfig: Config = {
    apiUrl: 'http://localhost:9000'
};

const prodConfig: Config = {
    apiUrl: 'https://scala-backend.onrender.com/' // Replace with your actual Render backend URL
};

export const config: Config = process.env.NODE_ENV === 'production' ? prodConfig : devConfig;
