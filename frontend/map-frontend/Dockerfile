# frontend/map-frontend/Dockerfile
# Stage 1: Build the React application
FROM node:20-alpine AS build
WORKDIR /app
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_TILE_URL=http://localhost:3000/ottawa_risk_baseline
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_TILE_URL=$VITE_TILE_URL
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine
# Copy the built assets from Stage 1 to Nginx's web directory
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
