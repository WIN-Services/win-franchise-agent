import React from 'react';
import { View, Text, StyleSheet, Linking, Platform } from 'react-native';

const HoverableLink = ({ text, url }) => {
  const [isHovered, setIsHovered] = React.useState(false);

  const webProps = Platform.OS === 'web' ? {
    onMouseEnter: () => setIsHovered(true),
    onMouseLeave: () => setIsHovered(false),
  } : {};

  return (
    <Text 
      style={[styles.linkText, isHovered && styles.linkHovered]} 
      onPress={() => Linking.openURL(url)}
      {...webProps}
    >
      {text}
      {isHovered && Platform.OS === 'web' && (
        <View style={styles.previewTooltip}>
          <Text style={styles.previewTooltipText} numberOfLines={1}>{url}</Text>
          <iframe 
             src={url} 
             style={{ width: 250, height: 150, border: 'none', marginTop: 4, borderRadius: 4, backgroundColor: '#fff' }} 
             title="preview"
          />
        </View>
      )}
    </Text>
  );
};

const ChatBubble = ({ message, isUser }) => {
  const formatMessage = (text) => {
    if (!text) return '';
    const regex = /\*\*([^\*]+)\*\*|\[([^\]]+)\]\(([^)]+)\)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      const matchIndex = match.index;

      if (matchIndex > lastIndex) {
        parts.push(text.substring(lastIndex, matchIndex));
      }

      if (match[1]) {
        // Bold text
        parts.push(
          <Text key={`bold-${matchIndex}`} style={styles.boldText}>
            {match[1]}
          </Text>
        );
      } else if (match[2] && match[3]) {
        // Markdown Link
        parts.push(
          <HoverableLink key={`link-${matchIndex}`} text={match[2]} url={match[3]} />
        );
      }

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  const parseMessageWithTables = (text) => {
    if (!text) return null;

    const lines = text.split('\n');
    const elements = [];
    let currentParagraph = [];
    let inTable = false;
    let tableHeaders = [];
    let tableRows = [];

    const flushParagraph = (keyPrefix) => {
      if (currentParagraph.length > 0) {
        const paragraphText = currentParagraph.join('\n');
        elements.push(
          <Text 
            key={`${keyPrefix}-${elements.length}`} 
            style={[
              styles.text,
              isUser ? styles.userText : styles.agentText
            ]}
          >
            {formatMessage(paragraphText)}
          </Text>
        );
        currentParagraph = [];
      }
    };

    const flushTable = (keyPrefix) => {
      if (tableHeaders.length > 0 || tableRows.length > 0) {
        elements.push(
          <View key={`table-${keyPrefix}-${elements.length}`} style={styles.tableContainer}>
            {/* Render Header */}
            {tableHeaders.length > 0 && (
              <View style={styles.tableHeaderRow}>
                {tableHeaders.map((header, index) => (
                  <View key={`header-cell-${index}`} style={[styles.tableHeaderCell, { flex: 1 }]}>
                    <Text style={styles.tableHeaderCellText}>
                      {formatMessage(header.trim())}
                    </Text>
                  </View>
                ))}
              </View>
            )}
            
            {/* Render Rows */}
            {tableRows.map((row, rowIndex) => (
              <View 
                key={`row-${rowIndex}`} 
                style={[
                  styles.tableRow, 
                  rowIndex % 2 === 1 ? styles.tableRowOdd : null,
                  rowIndex === tableRows.length - 1 ? styles.tableRowLast : null
                ]}
              >
                {row.map((cell, cellIndex) => (
                  <View key={`cell-${rowIndex}-${cellIndex}`} style={[styles.tableCell, { flex: 1 }]}>
                    <Text style={styles.tableCellText}>
                      {formatMessage(cell.trim())}
                    </Text>
                  </View>
                ))}
              </View>
            ))}
          </View>
        );
        tableHeaders = [];
        tableRows = [];
      }
      inTable = false;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const isTableRow = /^\s*\|.*\|\s*$/.test(line);

      if (isTableRow) {
        flushParagraph('para');
        inTable = true;

        const cells = line.split('|').map(c => c.trim());
        if (cells[0] === '') cells.shift();
        if (cells[cells.length - 1] === '') cells.pop();

        const isSeparator = cells.every(c => /^\s*:?-+:?\s*$/.test(c));

        if (isSeparator) {
          continue;
        }

        if (tableHeaders.length === 0 && tableRows.length === 0) {
          tableHeaders = cells;
        } else {
          tableRows.push(cells);
        }
      } else {
        if (inTable) {
          flushTable('table');
        }
        currentParagraph.push(line);
      }
    }

    if (inTable) {
      flushTable('table-end');
    } else {
      flushParagraph('para-end');
    }

    return elements;
  };

  return (
    <View style={[
      styles.container,
      isUser ? styles.userContainer : styles.agentContainer
    ]}>
      <View style={[
        styles.bubble,
        isUser ? styles.userBubble : styles.agentBubble
      ]}>
        {parseMessageWithTables(message)}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
    flexDirection: 'row',
    paddingHorizontal: 16,
  },
  userContainer: {
    justifyContent: 'flex-end',
  },
  agentContainer: {
    justifyContent: 'flex-start',
  },
  bubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 1,
  },
  userBubble: {
    backgroundColor: '#0056b3', // WIN Blue
    borderBottomRightRadius: 4,
  },
  agentBubble: {
    backgroundColor: '#f0f0f0',
    borderBottomLeftRadius: 4,
    zIndex: 1,
  },
  text: {
    fontSize: 16,
    lineHeight: 22,
  },
  boldText: {
    fontWeight: 'bold',
  },
  linkText: {
    color: '#0056b3',
    textDecorationLine: 'underline',
    fontWeight: '500',
  },
  linkHovered: {
    color: '#003d82',
  },
  previewTooltip: {
    position: 'absolute',
    bottom: 25,
    left: 0,
    width: 266,
    backgroundColor: '#333',
    padding: 8,
    borderRadius: 8,
    zIndex: 9999,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 10,
  },
  previewTooltipText: {
    color: '#fff',
    fontSize: 10,
    marginBottom: 2,
  },
  userText: {
    color: '#fff',
  },
  agentText: {
    color: '#333',
  },
  tableContainer: {
    marginVertical: 12,
    borderWidth: 1,
    borderColor: '#dee2e6',
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#fff',
    alignSelf: 'stretch',
  },
  tableHeaderRow: {
    flexDirection: 'row',
    backgroundColor: '#0056b3', // WIN Blue for headers
    borderBottomWidth: 1,
    borderBottomColor: '#dee2e6',
    paddingVertical: 8,
  },
  tableHeaderCell: {
    paddingHorizontal: 8,
    justifyContent: 'center',
  },
  tableHeaderCellText: {
    fontWeight: 'bold',
    color: '#fff',
    fontSize: 14,
  },
  tableRow: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    paddingVertical: 8,
    backgroundColor: '#fff',
  },
  tableRowOdd: {
    backgroundColor: '#f8f9fa',
  },
  tableRowLast: {
    borderBottomWidth: 0,
  },
  tableCell: {
    paddingHorizontal: 8,
    justifyContent: 'center',
  },
  tableCellText: {
    fontSize: 14,
    color: '#333',
  },
});

export default ChatBubble;
