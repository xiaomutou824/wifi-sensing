/*
 * ESP32-S3 A: SoftAP endpoint for CSI experiments.
 *
 * Keep this board powered on. The STA receiver pings the AP gateway
 * 192.168.4.1, and the AP replies produce stable AP-to-STA packets.
 */

#include <inttypes.h>
#include <string.h>

#include "esp_check.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#define MAC_STR "%02x:%02x:%02x:%02x:%02x:%02x"
#define MAC_ARG(mac) (mac)[0], (mac)[1], (mac)[2], (mac)[3], (mac)[4], (mac)[5]

static const char *TAG = "csi_ap";
static volatile uint32_t sta_connected_count;
static volatile uint32_t sta_disconnected_count;

static void init_nvs(void) {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);
}

static void ap_event_handler(void *arg, esp_event_base_t event_base,
                             int32_t event_id, void *event_data) {
  (void)arg;
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STACONNECTED) {
    wifi_event_ap_staconnected_t *event =
        (wifi_event_ap_staconnected_t *)event_data;
    sta_connected_count++;
    ESP_LOGI(TAG, "STA connected: " MAC_STR ", aid=%d", MAC_ARG(event->mac),
             event->aid);
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_AP_STADISCONNECTED) {
    wifi_event_ap_stadisconnected_t *event =
        (wifi_event_ap_stadisconnected_t *)event_data;
    sta_disconnected_count++;
    ESP_LOGW(TAG, "STA disconnected: " MAC_STR ", aid=%d", MAC_ARG(event->mac),
             event->aid);
  }
}

static esp_err_t wifi_init_ap(void) {
  ESP_RETURN_ON_ERROR(esp_netif_init(), TAG, "esp_netif_init failed");
  ESP_RETURN_ON_ERROR(esp_event_loop_create_default(), TAG,
                      "event loop init failed");
  esp_netif_create_default_wifi_ap();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_RETURN_ON_ERROR(esp_wifi_init(&cfg), TAG, "esp_wifi_init failed");
  ESP_RETURN_ON_ERROR(esp_event_handler_instance_register(
                          WIFI_EVENT, ESP_EVENT_ANY_ID, &ap_event_handler,
                          NULL, NULL),
                      TAG, "register AP event handler failed");

  wifi_config_t wifi_config = {0};
  strlcpy((char *)wifi_config.ap.ssid, CONFIG_CSI_AP_SSID,
          sizeof(wifi_config.ap.ssid));
  wifi_config.ap.ssid_len = strlen(CONFIG_CSI_AP_SSID);
  strlcpy((char *)wifi_config.ap.password, CONFIG_CSI_AP_PASSWORD,
          sizeof(wifi_config.ap.password));
  wifi_config.ap.channel = CONFIG_CSI_AP_CHANNEL;
  wifi_config.ap.max_connection = CONFIG_CSI_AP_MAX_STA_CONN;
  wifi_config.ap.authmode = strlen(CONFIG_CSI_AP_PASSWORD) == 0
                                ? WIFI_AUTH_OPEN
                                : WIFI_AUTH_WPA2_PSK;
  wifi_config.ap.pmf_cfg.required = false;

  ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_AP), TAG,
                      "set Wi-Fi AP mode failed");
  ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_AP, &wifi_config), TAG,
                      "set AP config failed");
  ESP_RETURN_ON_ERROR(esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20), TAG,
                      "set AP bandwidth failed");
  ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "start AP failed");

  uint8_t ap_mac[6] = {0};
  ESP_RETURN_ON_ERROR(esp_wifi_get_mac(WIFI_IF_AP, ap_mac), TAG,
                      "get AP MAC failed");
  ESP_LOGI(TAG, "SoftAP started: ssid=%s, channel=%d, auth=%s, bssid=" MAC_STR,
           CONFIG_CSI_AP_SSID, CONFIG_CSI_AP_CHANNEL,
           wifi_config.ap.authmode == WIFI_AUTH_OPEN ? "open" : "wpa2",
           MAC_ARG(ap_mac));
  ESP_LOGI(TAG, "Keep this AP board powered on. STA should ping 192.168.4.1");
  return ESP_OK;
}

static void ap_stats_task(void *arg) {
  (void)arg;
  while (true) {
    vTaskDelay(pdMS_TO_TICKS(CONFIG_CSI_STATS_INTERVAL_SEC * 1000));
    ESP_LOGI(TAG, "AP stats: sta_connected=%" PRIu32
                  ", sta_disconnected=%" PRIu32,
             sta_connected_count, sta_disconnected_count);
  }
}

void app_main(void) {
  init_nvs();
  ESP_LOGI(TAG, "ESP32-S3 role: AP endpoint");
  ESP_ERROR_CHECK(wifi_init_ap());
  xTaskCreate(ap_stats_task, "ap_stats", 3072, NULL, 2, NULL);
}
