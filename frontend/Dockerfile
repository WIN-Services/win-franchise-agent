# Stage 1: Build Expo App
FROM node:20-alpine AS build

WORKDIR /app

# Install package dependencies
COPY package*.json ./
# Running npm ci is faster and safer for consistent builds
RUN npm ci || npm install

# Copy the rest of the application source code
COPY . .

# Set production environment and bundle for web
ENV NODE_ENV=production
RUN npx expo export

# Stage 2: Serve via Nginx
FROM nginx:alpine

# Copy built files from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 9100
EXPOSE 9100

CMD ["nginx", "-g", "daemon off;"]
